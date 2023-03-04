from core.board import Board
import torch
import torch.nn.functional as F
import itertools
from analysis.utils import get_one_ind
import numpy as np

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
        d = MSM(self.bitmap - other.bitmap, self.n - other.n, self.size - other.size)
        d.pos = self.pos
        return d
    def __mul__(self, other):
        product = self.bitmap * other.bitmap
        d = MSM(product, min(self.n, other.n), torch.sum(product).item())
        d.pos = self.pos
        return d
    def __repr__(self):
        return repr(self.n) + ',' + repr(self.pos) + '=====\n' + repr(self.bitmap[0,0,...].numpy()) + '\n'

class MSM_Node:
    # Wrapper class for MSM to define relationships between other nearby MSM
    def __init__(self, msm):
        self.msm = msm
        self.witnesses = F.conv_transpose2d(msm.bitmap, WITNESS_KERNEL)[...,1:-1,1:-1]
        #self.vore_witnesses = self.witnesses == self.witnesses[...,msm.pos[0],msm.pos[1]] if msm.pos else None
        self.witnesses = self.witnesses != 0
        self.edges = {coord: dict() for coord in get_one_ind(self.witnesses[0,0,...])}
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
    def create_edge(self, other):
        if other.pos() in self.edges and self.pos() in other.edges:
            edge = MSM_Edge(self, other)
            if torch.sum(edge.intersection.bitmap) == 0: return
            self.edges[other.pos()][edge] = other
            other.edges[self.pos()][edge] = self
    def remove_edge(self, edge_info):
        edge, other = edge_info
        self.edges[other.pos()].pop(edge)
        if len(self.edges[other.pos()]) == 0:
            self.edges.pop(other.pos())
    def disconnect(self):
        # Remove self from edge list of neighbors
        for edge_coord in self.edges:
            for edge, other in self.edges[edge_coord].items():
                other.remove_edge((edge, self))
    def update_edges(self):
        # Update intersections and prune edges no longer intersecting
        for edge_coord in self.edges:
            to_remove = []
            for edge, other in self.edges[edge_coord].items():
                if edge.update(): to_remove.append((edge, other))
            for edge, other in to_remove: 
                self.edges[edge_coord].pop(edge)
                other.remove_edge((edge, self))


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

def get_msm_graph(B: Board, order=2):
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
    if suggestions := minecount_reduce_msm_graph(MSMs):
        to_clear = to_clear.union(suggestions[0])
        to_flag = to_flag.union(suggestions[1])
    prune_msm_graph_duplicates(MSMs)
    print(__flatten_msm_dict(MSMs), 'Initial Set ====================')

    # Find all second-order MSMs by partitioning MSMs by subsets
    if order >= 2:
        while expand_msm_graph_once(MSMs):
            print(__flatten_msm_dict(MSMs),'Expansion =======================================')
            prune_msm_graph_duplicates(MSMs)
            print(__flatten_msm_dict(MSMs),'Expansion Pruned =======================================')
            if suggestions := minecount_reduce_msm_graph(MSMs):
                to_clear = to_clear.union(suggestions[0])
                to_flag = to_flag.union(suggestions[1])
            print(__flatten_msm_dict(MSMs),'Reduced =======================================')
            prune_msm_graph_duplicates(MSMs)
            print(__flatten_msm_dict(MSMs),'End of Iteration =======================================')
    
    # Other handling stuffs
    __update_board(B_orig, to_clear, to_flag)
    for c_msm in MSMs[CKEY]:
        c_bm = c_msm.bitmap()
    if B.n == torch.sum(c_bm):
        to_flag = to_clear.union(get_one_ind(c_bm))
        MSMs.pop(CKEY)
        return B_orig, __flatten_msm_dict(MSMs), to_clear, to_flag
    if B.n == 0:
        to_clear = to_clear.union(get_one_ind(c_bm))
        MSMs.pop(CKEY)
        return B_orig, __flatten_msm_dict(MSMs), to_clear, to_flag
    return B_orig, __flatten_msm_dict(MSMs), to_clear, to_flag
    

def create_first_order_msm_graph(B):
    """
    Constructs a MSM graph from a minesweeper board based on the numbers on the board.
    Assumes board contains no known mines (i.e. is board reduced)
    Nodes are MSM regions.
    An edge exists between two nodes if they intersect (only nodes in radius 2 can intersect)
    """
    MSMs = dict()
    cmsm = MSM((1 - B.C)) # The trivial MSM created by the area not touching any number

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
        for d in itertools.product(range(-2,1), range(-2,3)):
            if d[0] == 0 and d[1] >= 0: continue
            dcoord = tuple(np.add(coord, d))
            if dcoord in MSMs:
                for other in MSMs[dcoord]:
                    m.create_edge(other)
        MSMs[coord] = set()
        MSMs[coord].add(m)

        # Update the complement msm by subtracting current bitmap
        cmsm.bitmap *= (1 - bitmap)

    # Add complement msm to dictionary
    MSMs[CKEY] = set()
    MSMs[CKEY].add(MSM_Node(cmsm))

    return MSMs
    

def prune_msm_graph_duplicates(MSMs):
    # Iterate over all MSMs in set
    to_remove = set()
    for coord in MSMs:
        for curr_node in MSMs[coord]:
            # Iterate over all edges of the node
            for edge_coord in curr_node.edges:
                for edge, other_node in curr_node.edges[edge_coord].items():
                    cm = curr_node.msm
                    om = other_node.msm
                    # If bitmaps are equal, remove the first one in the edge (by position)
                    if cm == om: 
                        print(cm, om, '923584750928374509283754098237049587230945872')
                        to_remove.add(edge.msm1)
    for dupe in to_remove:
        # Disconnect and remove node
        coord = dupe.pos()
        removed = set()
        # TODO bug with sets here? had to bandaid instead of calling remove
        #MSMs[coord].remove(dupe)
        while MSMs[coord] and (popped := MSMs[coord].pop()) != dupe:
            removed.add(popped)
        MSMs[coord] = MSMs[coord].union(removed)
        dupe.disconnect()
            


def minecount_reduce_msm_graph(MSMs):
    """
    Iterate over an MSM graph and remove nodes where either:
    - The number of mines is 0
    - The number of mines is equal to the size of the region (all mines)
    Remove them, and update their neighbors accordingly.
    Returns the coordinates of cells that should be cleared and flagged
    """
    to_clear = set()
    to_flag = set()
    # Repeat while a change occured
    minecounted = True
    while minecounted:
        minecounted = False
        # Iterate over all MSMs in set
        for coord in MSMs:
            to_remove = []
            for msm_node in MSMs[coord]:
                if msm_node.n() == 0:
                    # If the number is 0, it can be set to don't care and it's neighbors can be cleared
                    neighbors = get_one_ind(msm_node.bitmap()[0,0,...])
                    # Verify there is an effect
                    if len(neighbors) != 0: 
                        minecounted = True
                        to_clear = to_clear.union(neighbors)
                        to_update = []
                        # Iterate over edges to modify neighbors
                        for edge_coord in msm_node.edges:
                            for edge, other in msm_node.edges[edge_coord].items():
                                intersection = edge.intersection
                                diff = torch.sum(intersection.bitmap).item()
                                om = other.msm
                                # Shrink the other node and update edges
                                om.bitmap = om.bitmap * (1 - intersection.bitmap)
                                om.size -= diff
                                to_update.append(other)
                        for q in to_update:
                            q.update_edges()
                    to_remove.append(msm_node)

                elif msm_node.size() == msm_node.n():
                    # If the number is equal to the number of unknown cells, all the cells contain mines
                    neighbors = get_one_ind(msm_node.bitmap()[0,0,...])
                    # Verify there is an effect
                    if len(neighbors) != 0: 
                        minecounted = True
                        to_flag = to_flag.union(neighbors)
                        to_update = []
                        # Iterate over edges to modify neighbors
                        for edge_coord in msm_node.edges:
                            for edge, other in msm_node.edges[edge_coord].items():
                                intersection = edge.intersection
                                diff = torch.sum(intersection.bitmap).item()
                                om = other.msm
                                # Shrink the other node and update edges
                                om.bitmap = om.bitmap * (1 - intersection.bitmap)
                                om.n -= diff
                                om.size -= diff
                                to_update.append(other)
                        for q in to_update:
                            q.update_edges()
                    to_remove.append(msm_node)
            for counted in to_remove: 
                removed = set()
                # TODO bug with sets here? had to bandaid instead of calling remove
                #MSMs[coord].remove(counted)
                while MSMs[coord] and (popped := MSMs[coord].pop()) != counted:
                    removed.add(popped)
                MSMs[coord] = MSMs[coord].union(removed)
                counted.disconnect()
            if minecounted: break

    if len(to_clear)==0 and len(to_flag)==0: 
        return None # return None to indicate no change
    return to_clear, to_flag


def expand_msm_graph_once(MSMs):
    # Iterate over all MSMs in set
    to_add = set()
    for coord in MSMs:
        for curr_node in MSMs[coord]:
            # Iterate over all edges of the node
            for edge_coord in curr_node.edges:
                for edge, other_node in curr_node.edges[edge_coord].items():
                    cm = curr_node.msm
                    om = other_node.msm
                    im = edge.intersection

                    # Verify subset relations
                    if edge.intersection == cm:
                        dm = om - edge.intersection
                        dm_coord = dm.pos
                    elif edge.intersection == om:
                        dm = cm - edge.intersection
                        dm_coord = dm.pos
                    # Verify 1-2 relationship (TODO is there a better way?)
                    elif om.n - cm.n == om.size - im.size:
                        dm = om - edge.intersection
                        dm_coord = dm.pos
                    elif cm.n -om.n==cm.size-im.size:
                        dm = cm - edge.intersection
                        dm_coord = dm.pos
                    else: continue #otherwise continue on

                    # Create new node, and attempt to add later
                    dm_node = MSM_Node(dm)
                    to_add.add(dm_node)
    for new_node in to_add:
        for d in itertools.product(range(-2,3), range(-2,3)):
            dcoord = tuple(np.add(new_node.pos(), d))
            if dcoord == CKEY: continue
            if dcoord in MSMs:
                for other in MSMs[dcoord]:
                    new_node.create_edge(other)
        MSMs[new_node.pos()].add(new_node)
    return len(to_add) > 0
                    

def __update_board(B, to_clear, to_flag):
    for coord in to_clear:
        B.set_clear(*coord)
    for coord in to_flag:
        B.set_mine(*coord)

def __flatten_msm_dict(d):
    MSMs = set()
    for coord in d:
        msm_set = set(map(lambda node: node.msm, d[coord]))
        MSMs = MSMs.union(msm_set)
    return list(MSMs)

def __union2dset(s1, s2):
    for x in range(len(s1)):
        for y in range(len(s1[x])):
            s1[x][y] = s1[x][y].union(s2[x][y])