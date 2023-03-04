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
        return torch.all(self.bitmap == other.bitmap) and self.n == other.n
    def __eq__(self, other):
        return hash(self.bitmap)
    def __sub__(self, other):
        d = MSM(self.bitmap - other.bitmap, self.n - other.n, self.size - other.size)
        d.pos = self.pos
        return d
    def __mul__(self, other):
        d = MSM(self.bitmap * other.bitmap, min(self.n, other.n), min(self.size, other.size))
        d.pos = self.pos
        return d
    def __repr__(self):
        return repr(self.n) + '=====\n' + repr(self.bitmap[0,0,...].numpy()) + '\n'

class MSM_Node:
    # Wrapper class for MSM to define relationships between other nearby MSM
    def __init__(self, msm):
        self.msm = msm
        self.witnesses = F.conv_transpose2d(msm.bitmap, WITNESS_KERNEL)[...,1:-1,1:-1]
        self.vore_witnesses = self.witnesses == self.witnesses[...,msm.pos[0],msm.pos[1]] if msm.pos else None
        self.witnesses = self.witnesses != 0
        self.edges = {coord: set() for coord in get_one_ind(self.witnesses[0,0,...])}
    def __repr__(self):
        return repr(self.msm)
    def bitmap(self): return self.msm.bitmap
    def n(self): return self.msm.n
    def size(self): return self.msm.size
    def pos(self): return self.msm.pos
    def create_edge(self, other):
        if other.pos() in self.edges and self.pos() in other.edges:
            edge = MSM_Edge(self, other)
            self.edges[other.pos()].add((edge, other))
            other.edges[self.pos()].add((edge, self))
    def remove_edge(self, edge_info):
        edge, other = edge_info
        self.edges[other.pos()].remove(edge_info)
        if len(self.edges[other.pos()]) == 0:
            self.edges.pop(other.pos())
    def update_edges(self):
        # Update intersections and prune edges no longer intersecting
        for edge_coord in self.edges:
            to_remove = []
            for edge, other in self.edges[edge_coord]:
                if edge.update(): to_remove.append((edge, other))
            for r in to_remove: self.edges[edge_coord].remove(r)


class MSM_Edge:
    def __init__(self, msm1, msm2):
        self.msm1 = msm1
        self.msm2 = msm2
        self.intersection = msm1.bitmap() * msm2.bitmap()
    def update(self):
        self.intersection = self.msm1.bitmap() * self.msm2.bitmap()
        return torch.sum(self.intersection) == 0

def get_msm_graph(B: Board, order=2):
    assert order in [1,2]

    # mine reduce the board completely
    B = B.reduce()

    # Find all first-order MSMs induced by numbered cells
    MSMs = create_first_order_msm_graph(B)

    # If minecounting applies, we can mark certain coordinates as probability 1 areas
    to_clear = set()
    to_flag = set()
    
    # Iterate until no more minecounting patterns are found
    # Find all second-order MSMs by partitioning MSMs by subsets
    

    c_bm = MSMs[CKEY].bitmap
    if B.n == torch.sum(c_bm):
        to_flag = to_clear.union(get_one_ind(c_bm))
        MSMs.pop(CKEY)
        return B, list(__flatten_msm_dict(MSMs)), None, to_clear, to_flag
    if B.n == 0:
        to_clear = to_clear.union(get_one_ind(c_bm))
        MSMs.pop(CKEY)
        return B, list(__flatten_msm_dict(MSMs)), None, to_clear, to_flag
    return B, list(__flatten_msm_dict(MSMs)), to_clear, to_flag
    
def create_first_order_msm_graph(B):
    MSMs = dict()
    cmsm = MSM((1 - B.C)) # The trivial MSM created by the area not touching any number

    # Find the MSM induced by each number
    coords = get_one_ind(B.N[0,0,...] > -1)
    for coord in coords:
        # Get the bitmap and number of mines
        bitmap = (B.neighbor_mask(*coord) * (1 - B.C))
        mines = B[coord].N

        # Create the node and connect to edges backwards
        m = MSM_Node(MSM(bitmap, n=mines, pos=coord))
        # Coords are iterated lexicographically (see torch.nonzero)
        for d in itertools.product(range(-2,3), range(-2,1)):
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
    

def __flatten_msm_dict(d):
    MSMs = set()
    for coord in d:
        msm_set = map(lambda node: node.msm, d[coord])
        MSMs.union(msm_set)
    return MSMs

def __union2dset(s1, s2):
    for x in range(len(s1)):
        for y in range(len(s1[x])):
            s1[x][y] = s1[x][y].union(s2[x][y])

def minecount_reduce_msm_graph(MSMs):
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
                    to_remove.append(msm_node)
                    neighbors = get_one_ind(msm_node.bitmap()[0,0,...])
                    # Verify there is an effect
                    if len(neighbors) != 0: 
                        minecounted = True
                        to_clear = to_clear.union(neighbors)
                        # Iterate allow edges to modify neighbors
                        for edge_coord in msm_node.edges:
                            for edge, other in msm_node.edges[edge_coord]:
                                intersection = edge.intersection
                                diff = torch.sum(intersection).item()
                                om = other.msm
                                # Shrink the other node and update edges
                                om.bitmap = om.bitmap * (1 - intersection)
                                om.size -= diff
                                other.update_edges()

                elif msm_node.size() == msm_node.n():
                    # If the number is equal to the number of unknown cells, all the cells contain mines
                    to_remove.append(msm_node)
                    neighbors = get_one_ind(msm_node.bitmap()[0,0,...])
                    # Verify there is an effect
                    if len(neighbors) != 0: 
                        minecounted = True
                        to_flag = to_flag.union(neighbors)
                        # Iterate allow edges to modify neighbors
                        for edge_coord in msm_node.edges:
                            for edge, other in msm_node.edges[edge_coord]:
                                intersection = edge.intersection
                                diff = torch.sum(intersection).item()
                                om = other.msm
                                # Shrink the other node and update edges
                                om.bitmap = om.bitmap * (1 - intersection)
                                om.n -= diff
                                om.size -= diff
                                other.update_edges()
            for counted in to_remove: MSMs[coord].remove(counted)
            if minecounted: break

    if len(to_clear)==0 and len(to_flag)==0: 
        return None # return None to indicate no change
    return to_clear, to_flag
