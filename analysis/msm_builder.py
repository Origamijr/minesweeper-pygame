from core.board import Board
from core.bitmap import Bitmap
from analysis.utils import update_board
from analysis.msm_graph import MSM, MSM_Graph, MSM_Node

CKEY = (-1,-1)

def get_msm_graph(B: Board, order=2, reduced_board=False, verbose=0):
    assert order in [1,2]

    # mine reduce the board completely
    B_orig = B.clone()
    B = B.reduce()

    # If minecounting applies, we can mark certain coordinates as probability 1 areas
    to_clear = set()
    to_flag = set()

    # Find all first-order MSMs induced by numbered cells
    MSMG = create_first_order_msm_graph(B)

    # Find all second-order MSMs by partitioning MSMs by subsets and 1-2 rule
    if order >= 2:
        suggestions = second_order_msm_reduction(MSMG, minecount_first=True, verbose=verbose)
        to_clear = to_clear.union(suggestions[0])
        to_flag = to_flag.union(suggestions[1])
    
    # Other handling stuffs
    update_board(B_orig, to_clear, to_flag)
    update_board(B, to_clear, to_flag)
    B = B.reduce()
    if len(MSMG[CKEY]) == 0 or (c_bm := MSMG[CKEY][0].bitmap()).sum() == 0:
        MSMG.remove_node(MSMG[CKEY][0])
    elif B.n == c_bm.sum() and len(MSMG) == 1:
        to_flag = to_flag.union(set(c_bm.nonzero()))
        for coord in c_bm.nonzero(): 
            B.set_mine(*coord)
            B_orig.set_mine(*coord)
        MSMG.remove_node(MSMG[CKEY][0])
    elif B.n == 0:
        to_clear = to_clear.union(set(c_bm.nonzero()))
        for coord in c_bm.nonzero(): 
            B.set_clear(*coord)
            B_orig.set_clear(*coord)
        MSMG.remove_node(MSMG[CKEY][0])
    if verbose >= 2: print(f'Size of MSM Graph: {len(MSMG)}')
    return B if reduced_board else B_orig, MSMG, to_clear, to_flag
    

def create_first_order_msm_graph(B:Board):
    """
    Constructs a MSM graph from a minesweeper board based on the numbers on the board.
    Assumes board contains no known mines (i.e. is board reduced)
    Nodes are MSM regions.
    An edge exists between two nodes if they intersect (only nodes in radius 2 can intersect)
    """
    MSMG = MSM_Graph()
    cmsm_bitmap = -B.C # The trivial MSM created by the area not touching any number
    
    # Find the MSM induced by each number
    coords = (-B.N.get_mask(-1)).nonzero()
    for coord in coords:
        # Get the bitmap and number of mines
        bitmap = B.neighbor_mask(*coord) - B.C
        if bitmap.sum() == 0: continue # skip numbers with no unknowns
        mines = B[coord].N.item()

        # Create the node and connect to edges backwards
        MSMG.add_node(MSM_Node(MSM(bitmap, n=mines, pos=coord)))

        # Update the complement msm by subtracting current bitmap
        cmsm_bitmap -= bitmap

    # Add complement msm to dictionary
    MSMG.add_node(MSM_Node(MSM(cmsm_bitmap, pos=CKEY)))

    return MSMG
    


def second_order_msm_reduction(MSMG, minecount_first=False, verbose=0):
    # MSM reduction via repeated minecount reduction and graph extension

    to_clear = set()
    to_flag = set()

    if minecount_first:
        if suggestions := __minecount_reduce_msm_graph(MSMG, verbose=verbose):
            to_clear = to_clear.union(suggestions[0])
            to_flag = to_flag.union(suggestions[1])

    while __expand_msm_graph_once(MSMG, verbose=verbose):
        if suggestions := __minecount_reduce_msm_graph(MSMG, verbose=verbose):
            to_clear = to_clear.union(suggestions[0])
            to_flag = to_flag.union(suggestions[1])
    
    return to_clear, to_flag


def __minecount_reduce_msm_graph(MSMG: MSM_Graph, verbose=0):
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
        to_remove = []
        to_add = []
        # Iterate over all MSMs in set
        for msm_node in MSMG:
            # If the number is 0, its neighbors can be cleared
            if msm_node.n() == 0:
                neighbors = msm_node.bitmap().nonzero()
                # Verify there is an effect
                if len(neighbors) != 0: 
                    minecounted = True
                    to_clear = to_clear.union(neighbors)
                    # Iterate over edges to modify neighbors
                    for edge, other in msm_node.get_edges():
                        # Remove the other node too and add the difference
                        to_remove.append(other)
                        diff = other.msm - edge.intersection # subtract intersection
                        diff.n = other.n() # Should have same number of mines as before
                        if diff.size == 0: # Nothing to do if empty, but ensure validity
                            assert diff.n == 0
                            continue
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
                    for edge, other in msm_node.get_edges():
                        # Remove the other node too and add the difference
                        to_remove.append(other)
                        diff = other.msm - edge.intersection # subtract intersection
                        diff.n = other.n() - edge.intersection.size # decrement number of mines
                        if diff.size == 0: # Nothing to do if empty, but ensure validity
                            assert diff.n == 0
                            continue
                        to_add.append(MSM_Node(diff))
                to_remove.append(msm_node)
        
            # Remove and add the nodes that should be changed
            if verbose >= 4 and len(to_remove) > 0: print(f"to_remove: {to_remove}")
            if verbose >= 4 and len(to_add) > 0: print(f"to_add: {to_add}")
            for old_node in to_remove: 
                MSMG.remove_node(old_node)
            for new_node in to_add: 
                assert new_node.size() >= new_node.n() >= 0, to_add # Assert new node is valid
                MSMG.add_node(new_node)
                
            if minecounted: break
    
    if len(to_clear)==0 and len(to_flag)==0: 
        return None # return None to indicate no change
    return to_clear, to_flag


def __expand_msm_graph_once(MSMG:MSM_Graph, verbose=0):
    # Iterate over all MSMs in set
    if verbose >= 4: print('expansion ', '=' * 80)
    to_add = []
    for curr_node in MSMG:
        cm = curr_node.msm
        # Iterate over all independent subsets of edges of the node
        for inter_bitmap, inter_n, edges in __find_all_independent_neighbor_edges(curr_node):
            # If dealing with only one edge, just verify subset relations
            if len(edges) == 1:
                edge = edges[0][0]
                om = edges[0][1].msm
                if edge.intersection == om:
                    dm = cm - edge.intersection
                    dm.n = cm.n - om.n
                    if dm.size == 0: 
                        assert dm.n == 0
                        continue
                    to_add.append(MSM_Node(dm))
                    if verbose >= 4: print('found subset')
                    continue # both subset and 1-2 rule add difference, so no need to check for 1-2

            # Verify 1-2 relationship (TODO is there a better way?)
            diff = cm.bitmap - inter_bitmap
            diff_size = diff.sum()
            if cm.n - inter_n == diff_size:
                if diff_size > 0: 
                    dm = MSM(diff, n=cm.n-inter_n, size=diff_size, pos=cm.pos)
                    to_add.append(MSM_Node(dm))
                    if verbose >= 4: print(f'found 1-2 rule 1')
                for edge, other in edges:
                    dm = other.bitmap() - inter_bitmap
                    if dm.sum() == 0: continue
                    to_add.append(MSM_Node(MSM(dm, n=0, pos=other.pos())))
                    if verbose >= 4: print(f'found 1-2 rule 2')
            # TODO add rules 3 and 4
    added = False
    if verbose >= 4 and len(to_add) > 0: print(f"to_add: {to_add}")
    for new_node in to_add:
        assert new_node.size() >= new_node.n() >= 0, to_add # Assert new node is valid
        if MSMG.add_node(new_node): added = True
    if verbose >= 4 and len(to_add) > 0: print(f"added? {added}")
    return added # Return whether or not a new node was added
                    
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