from core.board import Board, NEIGHBOR_KERNEL
from analysis.msm_builder import get_msm_graph

class EP_Group_Node:
    def __init__(self):
        pass
    
class Number_Node:
    def __init__(self):
        pass

class EP_Graph:
    def __init__(self):
        self.numbers = []
        self.ep_groups = []
        pass

def get_ep_graph(B: Board, verbose=0):
    B, MSMs, _, _, = get_msm_graph(B, reduced_board=True, verbose=verbose)
    return msm_to_ep(B, MSMs)

def msm_to_ep(B: Board, MSMs: list, verbose=0):
    B = B.reduce()
    return MSMs