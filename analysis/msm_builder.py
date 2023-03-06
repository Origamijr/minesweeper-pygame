from core.board import Board
import torch
import torch.nn.functional as F
from collections import ChainMap
from analysis.utils import get_one_ind, flatten_msm_dict, update_board
import copy

WITNESS_KERNEL = torch.tensor([[[[1.,1.,1.],
                                 [1.,1.,1.],
                                 [1.,1.,1.]]]])

CKEY = (-1,-1)

class MSM:
    """
    Represents a Mutually Shared Mine (MSM) region:
    A set of coordinates such that it is known that exactly $n$ mines exists
    $n$ is None for aggregate MSM regions (e.g. the compelment set)
    - Bitmap : (1,1,r,c) Tensor
        contains a 1 at the coordinates included in the set
    - n : int
        number of mines in the region
    - pos : tuple
        coordinate of region if centered on a cell
    """
    def __init__(self, bitmap, n=None, pos=None, size=None):
        self.bitmap, self.n, self.pos= bitmap, n, pos
        self.size = torch.sum(self.bitmap).item() if size is None else size
    def __eq__(self, other):
        return torch.all(self.bitmap == other.bitmap)
    def __hash__(self):
        return hash(self.bitmap.numpy().tostring())
    def __sub__(self, other):
        return MSM(self.bitmap * (1 - other.bitmap), pos=self.pos)
    def __mul__(self, other):
        return MSM(self.bitmap * other.bitmap, pos=self.pos)
    def __repr__(self):
        return repr(self.n) + ' mine(s), ' + repr(self.pos) + ' ' + repr(int(self.size)) + '=====\n' + repr(self.bitmap[0,0,...].numpy()) + '\n'

class MSM_Node:
    # Wrapper class for MSM to define relationships between other nearby MSM
    def __init__(self, msm):
        self.msm = msm
        self.witnesses = F.conv_transpose2d(msm.bitmap, WITNESS_KERNEL)[...,1:-1,1:-1]
        #self.vore_witnesses = self.witnesses == self.witnesses[...,msm.pos[0],msm.pos[1]] if msm.pos else None
        self.witnesses = self.witnesses != 0
        self.edges = {coord: dict() for coord in get_one_ind(self.witnesses[0,0,...])}
        self.edge_list = None
    def __eq__(self, other):
        return self.msm == other.msm
    def __hash__(self):
        return hash(self.msm)
    def __repr__(self):
        return repr(self.msm)
    def bitmap(self): return self.msm.bitmap
    def n(self): return self.msm.n
    def size(self): return self.msm.size
    def pos(self): return self.msm.pos
    def get_edges(self):
        return [(edge, other) for edge, other in ChainMap(*[d for d in self.edges.values() if len(d) > 0]).items()]
    def create_edge(self, other):
        if other.pos() in self.edges and self.pos() in other.edges:
            edge = MSM_Edge(self, other)
            if torch.sum(edge.intersection.bitmap) == 0: return
            self.edges[other.pos()][edge] = other
            other.edges[self.pos()][edge] = self
        self.edge_list = None
    def remove_edge(self, edge_info):
        edge, other = edge_info
        if edge not in self.edges[other.pos()]: return 
        self.edges[other.pos()].pop(edge)
        self.edge_list = None
    def disconnect(self):
        # Remove self from edge list of neighbors
        for edge_coord in self.edges:
            for edge, other in self.edges[edge_coord].items():
                other.remove_edge((edge, self))
        self.edge_list = None
    def update_edges(self):
        # Update intersections and prune edges no longer intersecting
        for edge_coord in self.edges:
            to_remove = []
            for edge, other in self.edges[edge_coord].items():
                if edge.update(): to_remove.append((edge, other))
            for edge, other in to_remove: 
                self.edges[edge_coord].pop(edge)
                other.remove_edge((edge, self))
        self.edge_list = None


class MSM_Edge:
    # Contains some information about the intersection between nodes
    def __init__(self, msm1, msm2):
        # Enforce ordering, so duplicates can be effectively removed
        if msm1.pos()[0] > msm2.pos()[0] \
            or (msm1.pos()[0] == msm2.pos()[0] and msm1.pos()[1] >= msm2.pos()[1]):
            self.msm1 = msm1
            self.msm2 = msm2
        else:
            self.msm1 = msm2
            self.msm2 = msm1
        self.intersection = msm1.msm * msm2.msm
    def update(self):
        # Returns True if the nodes no longer intersect
        self.intersection = self.msm1.msm * self.msm2.msm
        return torch.sum(self.intersection.bitmap) == 0
        

def get_msm_graph(B: Board, order=2, flatten = False, verbose=0):
    assert order in [1,2]

    # mine reduce the board completely
    B_orig = B.copy()
    B = B.reduce()

    # If minecounting applies, we can mark certain coordinates as probability 1 areas
    to_clear = set()
    to_flag = set()

    # Find all first-order MSMs induced by numbered cells
    MSMs = create_first_order_msm_graph(B)

    # Prune graphs and remove minecountable regions
    prune_msm_graph_duplicates(MSMs)
    if suggestions := __minecount_reduce_msm_graph(MSMs):
        to_clear = to_clear.union(suggestions[0])
        to_flag = to_flag.union(suggestions[1])
    prune_msm_graph_duplicates(MSMs)

    # Find all second-order MSMs by partitioning MSMs by subsets and 1-2 rule
    if order >= 2:
        suggestions = second_order_msm_reduction(MSMs, verbose=verbose)
        to_clear = to_clear.union(suggestions[0])
        to_flag = to_flag.union(suggestions[1])
    
    # Other handling stuffs
    update_board(B_orig, to_clear, to_flag)
    update_board(B, to_clear, to_flag)
    B = B.reduce()
    if len(MSMs[CKEY]) == 0 or torch.sum((c_bm := MSMs[CKEY][0].bitmap())) == 0:
        MSMs.pop(CKEY)
    elif B.n == torch.sum(c_bm):
        to_flag = to_clear.union(get_one_ind(c_bm[0,0,...]))
        MSMs.pop(CKEY)
    elif B.n == 0:
        to_clear = to_clear.union(get_one_ind(c_bm[0,0,...]))
        MSMs.pop(CKEY)
    return B_orig, flatten_msm_dict(MSMs) if flatten else MSMs, to_clear, to_flag
    

def second_order_msm_reduction(MSMs, minecount_first = False, verbose=0):
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


def create_first_order_msm_graph(B):
    """
    Constructs a MSM graph from a minesweeper board based on the numbers on the board.
    Assumes board contains no known mines (i.e. is board reduced)
    Nodes are MSM regions.
    An edge exists between two nodes if they intersect (only nodes in radius 2 can intersect)
    """
    MSMs = dict()
    cmsm = MSM((1 - B.C), pos=CKEY) # The trivial MSM created by the area not touching any number

    # Find the MSM induced by each number
    coords = get_one_ind(B.N[0,0,...] > -1)
    coords.sort()
    for coord in coords:
        # Get the bitmap and number of mines
        bitmap = (B.neighbor_mask(*coord) * (1 - B.C))
        mines = B[coord].N.item()

        # Create the node and connect to edges backwards
        m = MSM_Node(MSM(bitmap, n=mines, pos=coord))
        # Coords are iterated lexicographically (see torch.nonzero)
        for dcoord in m.edges.keys():
            if dcoord not in MSMs: continue
            for other in MSMs[dcoord]:
                m.create_edge(other)
        MSMs[coord] = []
        MSMs[coord].append(m)

        # Update the complement msm by subtracting current bitmap
        cmsm.bitmap *= (1 - bitmap)

    # Add complement msm to dictionary
    MSMs[CKEY] = []
    MSMs[CKEY].append(MSM_Node(cmsm))

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
    if verbose >= 3: print('Mineccount reduce ', '=' * 80)
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
                    neighbors = get_one_ind(msm_node.bitmap()[0,0,...])
                    # Verify there is an effect
                    if len(neighbors) != 0: 
                        minecounted = True
                        to_clear = to_clear.union(neighbors)
                        # Iterate over edges to modify neighbors
                        for edge_coord in msm_node.edges:
                            for edge, other in msm_node.edges[edge_coord].items():
                                intersection = edge.intersection
                                diff = torch.sum(intersection.bitmap).item()
                                # Remove the other node too and add the difference
                                to_remove.append(other)
                                diff = other.msm - intersection
                                diff.n = other.n()
                                to_add.append(MSM_Node(diff))
                    to_remove.append(msm_node)

                # If the number is equal to the number of unknown cells, all the cells contain mines
                elif msm_node.size() == msm_node.n():
                    neighbors = get_one_ind(msm_node.bitmap()[0,0,...])
                    # Verify there is an effect
                    if len(neighbors) != 0: 
                        minecounted = True
                        to_flag = to_flag.union(neighbors)
                        # Iterate over edges to modify neighbors
                        for edge_coord in msm_node.edges:
                            for edge, other in msm_node.edges[edge_coord].items():
                                intersection = edge.intersection
                                diff = torch.sum(intersection.bitmap).item()
                                # Remove the other node too and add the difference
                                to_remove.append(other)
                                diff = other.msm - intersection
                                diff.n = other.n() - intersection.size
                                to_add.append(MSM_Node(diff))
                    to_remove.append(msm_node)
            
            # Remove and add the nodes that should be changed
            if verbose >= 3 and len(to_remove) > 0: print(f"to_remove: {to_remove}")
            if verbose >= 3 and len(to_add) > 0: print(f"to_add: {to_add}")
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
    if verbose >= 3: print('expansion ', '=' * 80)
    to_add = []
    for coord in MSMs:
        for curr_node in MSMs[coord]:
            cm = curr_node.msm
            # Iterate over all independent subsets of edges of the node
            for inter_bitmap, inter_n, edges in __find_all_independent_neighbor_edges(curr_node):
                # If dealing with only one edge, just verify subset relations
                if len(edges) == 1:
                    edge = edges[0][0]
                    om = edges[0][1].msm
                    if edge.intersection == om:
                        if verbose >= 3: print('found subset')
                        dm = cm - edge.intersection
                        dm.n = cm.n - om.n
                        to_add.append(MSM_Node(dm))
                        continue # both subset and 1-2 rule add difference, so no need to check for 1-2

                # Verify 1-2 relationship (TODO is there a better way?)
                diff = cm.bitmap - inter_bitmap
                diff_size = int(torch.sum(diff))
                if diff_size > 0 and cm.n - inter_n == diff_size:
                    if verbose >= 3: print(f'found 1-2 rule')
                    dm = MSM(diff, n=cm.n-inter_n, size=diff_size, pos=cm.pos)
                    to_add.append(MSM_Node(dm))
            """
            for edge_coord in curr_node.edges:
                for edge, other_node in curr_node.edges[edge_coord].items():
                    cm = curr_node.msm
                    om = other_node.msm
                    im = edge.intersection

                    # Verify subset relations
                    if edge.intersection == cm:
                        dm = om - edge.intersection
                        dm.n = om.n - cm.n
                    elif edge.intersection == om:
                        dm = cm - edge.intersection
                        dm.n = cm.n - om.n
                    # Verify 1-2 relationship (TODO is there a better way?)
                    elif om.n - cm.n == om.size - im.size:
                        dm = om - edge.intersection
                        dm.n = om.n - cm.n
                    elif cm.n - om.n == cm.size - im.size:
                        dm = cm - edge.intersection
                        dm.n = cm.n - om.n
                    else: continue #otherwise continue on

                    # Create new node, and attempt to add later
                    dm_node = MSM_Node(dm)
                    to_add.append(dm_node)
            """
    added = False
    if verbose >= 3 and len(to_add) > 0: print(f"to_add: {to_add}")
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
    if verbose >= 3 and len(to_add) > 0: print(f"added? {added}")
    return added
                    
def __find_all_independent_neighbor_edges(node: MSM_Node):
    def __find_all_independent_neighbor_edges_helper(bitmap, edges):
        if not edges: return []
        ind_neighbors = []
        while edges:
            edge, other = edges.pop()
            union_bitmap = bitmap + edge.intersection.bitmap
            if not torch.any(union_bitmap == 2):
                n = int(other.n())
                ind_neighbors.append((union_bitmap, n, [(edge, other)]))
                ind_others = __find_all_independent_neighbor_edges_helper(union_bitmap, edges.copy())
                for i, (o_bitmap, o_n, others) in enumerate(ind_others):
                    ind_neighbors.append((o_bitmap, n+o_n ,others+[(edge, other)]))
        return ind_neighbors
                
    return __find_all_independent_neighbor_edges_helper(torch.zeros(node.bitmap().shape), list(node.get_edges()))