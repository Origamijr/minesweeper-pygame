import torch
from analysis.utils import flatten_msm_dict
from analysis.msm_builder import MSM, MSM_Node, second_order_msm_reduction
import copy
from math import comb
import itertools


def seperate_connected_msm_components(MSMs):
    # This is just dfs
    flat_graph = flatten_msm_dict(MSMs)
    assert len(flat_graph) > 0
    if len(flat_graph) == 1: return [MSMs]
    components = []
    visited = []
    while len(visited) != len(flat_graph):
        start = None
        for start in flat_graph: 
            if start not in visited: break
        stack = [start]
        visited.append(start)
        component = dict()
        component[start.pos()] = []
        component[start.pos()].append(start)
        while stack:
            curr_node = stack.pop()
            neighbors = [other for _, other in curr_node.get_edges() if other not in visited]
            if neighbors:
                visited += neighbors
                stack += neighbors
                for n in neighbors:
                    if n.pos() not in component: component[n.pos()] = []
                    component[n.pos()].append(n)
        components.append(component)
    return components

def find_num_solutions(MSMs, min_n=None, max_n=None, seed=None, verbose=0):
    """
    Find solutions using branch and bound. Faster if connected component.
    """
    counts = dict()
    flat_graph = flatten_msm_dict(MSMs)

    if len(flat_graph) == 0:
        counts[0] = 1
        return counts
    
    # If one MSM, the count is just n choose k where n is the size and k is the number of mines
    if len(flat_graph) == 1:
        size = int(flat_graph[0].size())
        if flat_graph[0].n() is not None:
            n = int(flat_graph[0].n())
            if (n < 0): print(flat_graph)
            counts[n] = comb(size, n)
            return counts
        if min_n == None: min_n = 0
        if max_n == None or max_n > size: max_n = size
        for n in range(min_n, max_n+1):
            counts[n] = comb(size, n)
        return counts
    
    # Otherwise, do branch and bound
    # TODO, I think there's a way to use connected components before branching and bounding.
    # Select arbitrary coordinate covered by one MSM
    if seed: torch.manual_seed(seed)
    random_bitmap = flat_graph[0 if True else torch.randint(len(flat_graph),(1,)).item()]
    random_candidates = random_bitmap.bitmap().nonzero()
    #if len(random_candidates) == 0: print(flat_graph)
    coord = tuple(random_candidates[0 if True else torch.randint(len(random_candidates),(1,))].numpy().flatten())
    bitmap = torch.zeros(random_bitmap.bitmap().shape)
    bitmap[0,0,coord[2],coord[3]] = 1

    # Count the cases if a mine is present at the selected coordinate
    if verbose >= 3: print(f'Try mine at {coord}')
    MSMs_mine, to_clear_m, to_flag_m = try_step_msm(MSMs, mine_bitmap=bitmap, verbose=verbose)
    num_mines = len(to_flag_m)
    num_with_mine = find_num_solutions(MSMs_mine, min_n=min_n, max_n=max_n, seed=seed, verbose=verbose)
    for n, count in num_with_mine.items():
        if n+num_mines not in counts: counts[n+num_mines] = 0
        counts[n+num_mines] += count
    
    # Count the cases if a mine is not present at the selected coordinate
    if verbose >= 3: print(f'Try clear at {coord}')
    MSMs_clear, to_clear_c, to_flag_c = try_step_msm(MSMs, clear_bitmap=bitmap, verbose=verbose)
    num_mines = len(to_flag_c)
    num_wo_mine = find_num_solutions(MSMs_clear, min_n=min_n, max_n=max_n, seed=seed, verbose=verbose)
    for n, count in num_wo_mine.items():
        if n+num_mines not in counts: counts[n+num_mines] = 0
        counts[n+num_mines] += count

    return counts
    

def try_step_msm(MSMs, clear_bitmap=None, mine_bitmap=None, verbose=0):
    # Assert nonoverlapping bitmaps
    assert (clear_bitmap is None or mine_bitmap is None) or torch.sum(mine_bitmap * clear_bitmap) == 0
    MSMs = copy.deepcopy(MSMs) # TODO find a better way to copy

    # Create MSMs based on the bitmaps and insert them into the graph
    if clear_bitmap is not None:
        clear_bitmap = clear_bitmap.clone()
        # Set position of bitmap to arbitrary location in graph (if diameter>2, may cause errors)
        pos = tuple(clear_bitmap[0,0,...].nonzero()[0].numpy())
        num_clear = torch.sum(clear_bitmap).item()
        # Create a Node for the cleared positions with 0 mines
        clear_node = MSM_Node(MSM(clear_bitmap, n=0, pos=pos, size=num_clear))
        # Insert node into graph
        for dcoord in clear_node.edges.keys():
            if dcoord not in MSMs: continue
            for other in MSMs[dcoord]:
                clear_node.create_edge(other)
        if pos not in MSMs: MSMs[pos] = []
        MSMs[pos].append(clear_node)

    # Identical to above
    if mine_bitmap is not None:
        mine_bitmap = mine_bitmap.clone()
        # Set position of bitmap to arbitrary location in graph (if diameter>2, may cause errors)
        pos = tuple(mine_bitmap[0,0,...].nonzero()[0].numpy())
        num_mine = torch.sum(mine_bitmap).item()
        # Create a Node for the cleared positions with all mines
        mine_node = MSM_Node(MSM(mine_bitmap, n=num_mine, pos=pos, size=num_mine))
        # Insert node into graph
        for dcoord in mine_node.edges.keys():
            if dcoord not in MSMs: continue
            for other in MSMs[dcoord]:
                mine_node.create_edge(other)
        if pos not in MSMs: MSMs[pos] = []
        MSMs[pos].append(mine_node)

    # Solve the graph with the new node inserted
    to_clear, to_flag = second_order_msm_reduction(MSMs, minecount_first=True, verbose=verbose)

    return MSMs, to_clear, to_flag


def quine_mcclusky_subset():
    # Would finding a subset cover with minimal overlap help?
    pass