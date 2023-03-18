from core.board import Board
from core.bitmap import Bitmap
import torch
import torch.nn.functional as F
from collections import ChainMap
from analysis.utils import get_one_ind, flatten_msm_dict, update_board
from analysis.msm_graph import MSM, MSM_Edge, MSM_Node

CKEY = (-1,-1)

def get_msm_graph(B: Board, order=2, flatten = False, reduced_board=False, verbose=0):
    assert order in [1,2]

    # mine reduce the board completely
    B_orig = B.copy()
    B = B.reduce()

    # If minecounting applies, we can mark certain coordinates as probability 1 areas
    to_clear = set()
    to_flag = set()

    # Find all first-order MSMs induced by numbered cells
    MSMs = create_first_order_msm_graph(B)

    # Find all second-order MSMs by partitioning MSMs by subsets and 1-2 rule
    if order >= 2:
        prune_msm_graph_duplicates(MSMs)

        suggestions = second_order_msm_reduction(MSMs, minecount_first=True, verbose=verbose)
        to_clear = to_clear.union(suggestions[0])
        to_flag = to_flag.union(suggestions[1])
    
    # Other handling stuffs
    update_board(B_orig, to_clear, to_flag)
    update_board(B, to_clear, to_flag)
    B = B.reduce()
    if len(MSMs[CKEY]) == 0 or (c_bm := MSMs[CKEY][0].bitmap()).sum() == 0:
        MSMs.pop(CKEY)
    elif B.n == c_bm.sum():
        to_flag = to_flag.union(set(c_bm.nonzero()))
        MSMs.pop(CKEY)
    elif B.n == 0:
        to_clear = to_clear.union(set(c_bm.nonzero()))
        MSMs.pop(CKEY)
    return B if reduced_board else B_orig, flatten_msm_dict(MSMs) if flatten else MSMs, to_clear, to_flag
    

def second_order_msm_reduction(MSMs, minecount_first=False, verbose=0):
    # MSM reduction via repeated minecount reduction and graph extension

    to_clear = set()
    to_flag = set()

    if minecount_first:
        if suggestions := __minecount_reduce_msm_graph(MSMs, verbose=verbose):
            to_clear = to_clear.union(suggestions[0])
            to_flag = to_flag.union(suggestions[1])
        prune_msm_graph_duplicates(MSMs, verbose=verbose)

    while __expand_msm_graph_once(MSMs, verbose=verbose):
        if suggestions := __minecount_reduce_msm_graph(MSMs, verbose=verbose):
            to_clear = to_clear.union(suggestions[0])
            to_flag = to_flag.union(suggestions[1])
        prune_msm_graph_duplicates(MSMs, verbose=verbose)
    
    return to_clear, to_flag


def create_first_order_msm_graph(B:Board):
    """
    Constructs a MSM graph from a minesweeper board based on the numbers on the board.
    Assumes board contains no known mines (i.e. is board reduced)
    Nodes are MSM regions.
    An edge exists between two nodes if they intersect (only nodes in radius 2 can intersect)
    """
    MSMs = dict()
    cmsm_bitmap = -B.C # The trivial MSM created by the area not touching any number
    
    # Find the MSM induced by each number
    coords = (-B.N.get_mask(-1)).nonzero()
    for coord in coords:
        # Get the bitmap and number of mines
        bitmap = B.neighbor_mask(*coord) - B.C
        if bitmap.sum() == 0: continue # skip numbers with no unknowns
        mines = B[coord].N.item()

        # Create the node and connect to edges backwards
        m = MSM_Node(MSM(bitmap, n=mines, pos=coord))
        # Coords are iterated lexicographically
        for dcoord in m.edges.keys():
            if dcoord not in MSMs: continue
            for other in MSMs[dcoord]:
                m.create_edge(other)
        MSMs[coord] = []
        MSMs[coord].append(m)

        # Update the complement msm by subtracting current bitmap
        cmsm_bitmap -= bitmap

    # Add complement msm to dictionary
    MSMs[CKEY] = []
    MSMs[CKEY].append(MSM_Node(MSM(cmsm_bitmap, pos=CKEY)))

    return MSMs
    

def prune_msm_graph_duplicates(MSMs, verbose=0):
    # Iterate over all MSMs in set
    to_remove = []
    to_keep_one = []
    for coord in MSMs:
        for curr_node in MSMs[coord]:
            if curr_node.size() == 0: to_remove.append(curr_node) # TODO bug, there shouldn't be any empty nodes
            # Iterate over all edges of the node
            for edge_coord in curr_node.edges:
                for edge, other_node in curr_node.edges[edge_coord].items():
                    cm = curr_node.msm
                    om = other_node.msm
                    # If bitmaps are equal, remove the first one in the edge (by position)
                    if cm == om:
                        to_remove.append(edge.msm1)
                        if cm.pos == om.pos:
                            to_keep_one.append(edge.msm1)
    for dupe in to_remove:
        # Disconnect and remove node
        coord = dupe.pos()
        removed = []
        # TODO bug with sets here? had to bandaid instead of calling remove
        #MSMs[coord].remove(dupe)
        while MSMs[coord] and (popped := MSMs[coord].pop()) != dupe:
            removed.append(popped)
        if dupe in to_keep_one:
            while MSMs[coord] and (popped := MSMs[coord].pop()) != dupe:
                removed.append(popped)
            MSMs[coord] += removed
            MSMs[coord].append(dupe)
            if not MSMs[coord] and popped != dupe: continue
        MSMs[coord] += removed
        dupe.disconnect()
            


def __minecount_reduce_msm_graph(MSMs, verbose=0):
    """
    Iterate over an MSM graph and remove nodes where either:
    - The number of mines is 0
    - The number of mines is equal to the size of the region (all mines)
    Remove them, and update their neighbors accordingly.
    Returns the coordinates of cells that should be cleared and flagged
    """
    if verbose >= 4: print('Mineccount reduce ', '=' * 80)
    to_clear = set()
    to_flag = set()
    # Repeat while a change occured
    minecounted = True
    while minecounted:
        minecounted = False
        # Iterate over all MSMs in set
        for coord in MSMs:
            to_remove = []
            to_add = []
            for msm_node in MSMs[coord]:
                # If the number is 0, its neighbors can be cleared
                if msm_node.n() == 0:
                    neighbors = msm_node.bitmap().nonzero()
                    # Verify there is an effect
                    if len(neighbors) != 0: 
                        minecounted = True
                        to_clear = to_clear.union(neighbors)
                        # Iterate over edges to modify neighbors
                        for edge_coord in msm_node.edges:
                            for edge, other in msm_node.edges[edge_coord].items():
                                intersection = edge.intersection
                                diff = intersection.bitmap.sum()
                                # Remove the other node too and add the difference
                                to_remove.append(other)
                                diff = other.msm - intersection
                                diff.n = other.n()
                                to_add.append(MSM_Node(diff))
                    to_remove.append(msm_node)

                # If the number is equal to the number of unknown cells, all the cells contain mines
                elif msm_node.size() == msm_node.n():
                    neighbors = msm_node.bitmap().nonzero()
                    # Verify there is an effect
                    if len(neighbors) != 0: 
                        minecounted = True
                        to_flag = to_flag.union(neighbors)
                        # Iterate over edges to modify neighbors
                        for edge_coord in msm_node.edges:
                            for edge, other in msm_node.edges[edge_coord].items():
                                intersection = edge.intersection
                                diff = intersection.bitmap.sum()
                                # Remove the other node too and add the difference
                                to_remove.append(other)
                                diff = other.msm - intersection
                                diff.n = other.n() - intersection.size
                                to_add.append(MSM_Node(diff))
                    to_remove.append(msm_node)
            
            # Remove and add the nodes that should be changed
            if verbose >= 4 and len(to_remove) > 0: print(f"to_remove: {to_remove}")
            if verbose >= 4 and len(to_add) > 0: print(f"to_add: {to_add}")
            for counted in to_remove: 
                removed = []
                pos = counted.pos()
                # TODO bug with sets here? had to bandaid instead of calling remove
                #MSMs[coord].remove(counted)
                while MSMs[pos] and (popped := MSMs[pos].pop()) != counted:
                    removed.append(popped)
                MSMs[pos] += removed
                counted.disconnect()
            for shrunk in to_add: 
                # Skip node if bitmap already in node TODO make more efficient
                assert shrunk.size() >= shrunk.n() >= 0, to_add
                dupe = False
                for other in flatten_msm_dict(MSMs):
                    if other == shrunk: 
                        dupe = True
                        break
                if dupe: continue
                # add to graph
                for dcoord in shrunk.edges.keys():
                    if dcoord not in MSMs: continue
                    for other in MSMs[dcoord]:
                        shrunk.create_edge(other)
                MSMs[shrunk.pos()].append(shrunk)
            if minecounted: break

    if len(to_clear)==0 and len(to_flag)==0: 
        return None # return None to indicate no change
    return to_clear, to_flag


def __expand_msm_graph_once(MSMs, verbose=0):
    # Iterate over all MSMs in set
    if verbose >= 4: print('expansion ', '=' * 80)
    to_add = []
    for coord in MSMs:
        for curr_node in MSMs[coord]:
            cm = curr_node.msm
            #if verbose >= 3: print(cm, curr_node.edges)
            # Iterate over all independent subsets of edges of the node
            for inter_bitmap, inter_n, edges in __find_all_independent_neighbor_edges(curr_node):
                #if verbose >= 3: print(inter_bitmap, inter_n, edges)
                # If dealing with only one edge, just verify subset relations
                if len(edges) == 1:
                    edge = edges[0][0]
                    om = edges[0][1].msm
                    if edge.intersection == om:
                        if verbose >= 4: print('found subset')
                        dm = cm - edge.intersection
                        dm.n = cm.n - om.n
                        to_add.append(MSM_Node(dm))
                        continue # both subset and 1-2 rule add difference, so no need to check for 1-2

                # Verify 1-2 relationship (TODO is there a better way?)
                diff = cm.bitmap - inter_bitmap
                diff_size = diff.sum()
                if diff_size > 0 and cm.n - inter_n == diff_size:
                    if verbose >= 4: print(f'found 1-2 rule')
                    dm = MSM(diff, n=cm.n-inter_n, size=diff_size, pos=cm.pos)
                    to_add.append(MSM_Node(dm))
                    for edge, other in edges:
                        to_add.append(MSM_Node(MSM(other.bitmap() - inter_bitmap, n=0, pos=other.pos())))
    added = False
    if verbose >= 4 and len(to_add) > 0: print(f"to_add: {to_add}")
    for new_node in to_add:
        # Skip node if bitmap already in node TODO make more efficient
        assert new_node.size() >= new_node.n() >= 0, to_add
        dupe = False
        for other in flatten_msm_dict(MSMs):
            if other == new_node: 
                dupe = True
                break
        if dupe: continue
        added = True
        # add to graph
        for dcoord in new_node.edges.keys():
            if dcoord not in MSMs: continue
            for other in MSMs[dcoord]:
                new_node.create_edge(other)
        MSMs[new_node.pos()].append(new_node)
    if verbose >= 4 and len(to_add) > 0: print(f"added? {added}")
    return added
                    
def __find_all_independent_neighbor_edges(node: MSM_Node):
    def __find_all_independent_neighbor_edges_helper(bitmap:Bitmap, edges):
        if not edges: return []
        ind_neighbors = []
        while edges:
            edge, other = edges.pop()
            union_bitmap = bitmap + edge.intersection.bitmap
            if not (bitmap * edge.intersection.bitmap).any():
                n = int(other.n())
                ind_neighbors.append((union_bitmap, n, [(edge, other)]))
                ind_others = __find_all_independent_neighbor_edges_helper(union_bitmap, edges.copy())
                for i, (o_bitmap, o_n, others) in enumerate(ind_others):
                    ind_neighbors.append((o_bitmap, n+o_n ,others+[(edge, other)]))
        return ind_neighbors
    rows, cols = node.bitmap().rows, node.bitmap().cols
    return __find_all_independent_neighbor_edges_helper(Bitmap(rows, cols), list(node.get_edges()))