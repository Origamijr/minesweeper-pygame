from core.bitmap import Bitmap
from analysis.utils import get_one_ind
from collections import ChainMap

class MSM:
    """
    Represents a Mutually Shared Mine (MSM) region:
    A set of coordinates such that it is known that exactly $n$ mines exists
    $n$ is None for aggregate MSM regions (e.g. the compelment set)
    - Bitmap : (1,1,r,c) Tensor
        contains a 1 at the coordinates included in the set
    - n : int
        number of mines in the region if known
    - pos : tuple
        coordinate of region if centered on a cell
    """
    def __init__(self, bitmap:Bitmap, n=None, pos=None, size=None):
        self.bitmap, self.n, self.pos= bitmap, n, pos
        self.size = self.bitmap.sum() if size is None else size
    def __eq__(self, other):
        return self.bitmap == other.bitmap
    def __hash__(self):
        return hash(self.bitmap)
    def __sub__(self, other):
        return MSM(self.bitmap - other.bitmap, pos=self.pos)
    def __mul__(self, other):
        return MSM(self.bitmap * other.bitmap, pos=self.pos)
    def __repr__(self):
        return repr(self.n) + ' mine(s), ' + repr(self.pos) + ' ' + repr(int(self.size)) + '=====\n' + repr(self.bitmap) + '\n'

class MSM_Node:
    # Wrapper class for MSM to define relationships between other nearby MSM
    def __init__(self, msm:MSM):
        self.msm = msm
        self.edges = {coord: dict() for coord in self.msm.bitmap.closure().nonzero()}
        self.edge_list = None
    def __eq__(self, other):
        return self.msm == other.msm
    def __hash__(self):
        return hash(self.msm)
    def __repr__(self):
        return repr(self.msm)
    def bitmap(self): return hash(self.msm)
    def n(self): return self.msm.n
    def size(self): return self.msm.size
    def pos(self): return self.msm.pos
    def get_edges(self):
        return [(edge, other) for edge, other in ChainMap(*[d for d in self.edges.values() if len(d) > 0]).items()]
    def create_edge(self, other):
        if other.pos() in self.edges and self.pos() in other.edges:
            edge = MSM_Edge(self, other)
            if edge.intersection.bitmap.sum() == 0: return # connect if intersect
            self.edges[other.pos()][edge] = other
            other.edges[self.pos()][edge] = self
        self.edge_list = None
    def remove_edge(self, edge_info):
        edge, other = edge_info
        if edge not in self.edges[other.pos()]: return 
        self.edges[other.pos()].pop(edge) # Only disconnects one way
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
    def __init__(self, msm1:MSM_Node, msm2:MSM_Node):
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
        return self.intersection.bitmap.sum() == 0
    

class MSM_Graph:
    def __init__(self):
        self.MSMs = dict()
    def __contains__(self, item):
        if isinstance(item, tuple): return item in self.MSMs
        if not isinstance(item, MSM_Node): return False
        for dcoord in item.edges.keys():
            if dcoord not in self.MSMs: continue
            if item in self.MSMs[dcoord]: return True
        return item in self.MSMs[item.pos()]
    def add_node(self, node:MSM_Node):
        if node in self.MSMs: return
        coord = node.pos()
        self.MSMs[coord] = []
        for dcoord in node.edges.keys():
            if dcoord not in self.MSMs: continue
            for other in self.MSMs[dcoord]:
                node.create_edge(other)
        if coord not in self.MSMs: self.MSMs[coord] = []
        self.MSMs[coord].append(node)
    def remove_node(self, node:MSM_Node):
        self.MSMs[node.pos()].remove(node)
        node.disconnect()
    def flatten(self):
        MSMs = []
        for coord in self.MSMs:
            MSMs += self.MSMs[coord]
        return MSMs
