import torch
from core.bitmap import Bitmap
from analysis.msm_builder import MSM, MSM_Node, MSM_Graph, second_order_msm_reduction
from analysis.solution_set import SolutionSet
from solver.solver_stats import SolverStats
from math import comb
from itertools import combinations
from functools import reduce

def seperate_connected_msm_components(MSMG:MSM_Graph) -> list[MSM_Graph]:
    # This is just dfs, but can maybe consider doing union forest instead if performance is an issue
    assert len(MSMG) > 0
    if len(MSMG) == 1: return [MSMG]
    components = []
    visited = []
    while len(visited) != len(MSMG):
        start = None
        # Find next unvisited node
        for start in MSMG:
            if start not in visited: break
        stack = [start]
        visited.append(start)

        # Build component from neighbors of start node
        component = MSM_Graph()
        component.add_node(start, connect_edges=False)
        while stack:
            curr_node = stack.pop()
            neighbors = [other for _, other in curr_node.get_edges() if other not in visited]
            if neighbors:
                visited += neighbors
                stack += neighbors
                for n in neighbors:
                    component.add_node(n, connect_edges=False)
        components.append(component)

    # Note: The components have the same references as the original graph
    return components

def try_step_msm(MSMG:MSM_Graph, clear_bitmap:Bitmap=None, mine_bitmap:Bitmap=None, number_bitmap:Bitmap=None, verbose=0):
    """
    Logic reduce a MSM graph with a set of spaces to assume clear and/or mine
    Does not guarantee consistency is maintained
    """

    # Assert nonoverlapping bitmaps
    assert (clear_bitmap is None or mine_bitmap is None) or not (mine_bitmap * clear_bitmap).any()
    MSMG = MSMG.clone() # get copy as to not modify original graph

    # Create MSMs based on the bitmaps and insert them into the graph
    if clear_bitmap is not None:
        clear_bitmap = clear_bitmap.clone()
        # Set position of bitmap to arbitrary location in graph (if diameter>2, may cause errors)
        pos = clear_bitmap.nonzero()[0]
        num_clear = clear_bitmap.sum()
        # Create a Node for the cleared positions with 0 mines
        MSMG.add_node(MSM_Node(MSM(clear_bitmap, n=0, pos=pos, size=num_clear)))
        
    # Identical to above
    if mine_bitmap is not None:
        mine_bitmap = mine_bitmap.clone()
        # Set position of bitmap to arbitrary location in graph (if diameter>2, may cause errors)
        pos = mine_bitmap.nonzero()[0]
        num_mine = mine_bitmap.sum()
        # Create a Node for the cleared positions with all mines
        MSMG.add_node(MSM_Node(MSM(mine_bitmap, n=num_mine, pos=pos, size=num_mine)))

    # Same, but with a number creating an MSM over the neighbors
    if number_bitmap is not None:
        number_bitmap = number_bitmap.clone()
        for pos, dec_bit in zip(number_bitmap.nonzero(), number_bitmap.decimate()):
            MSMG.add_node(MSM_Node(MSM(dec_bit.closure() * MSMG.bitmap(), n=number_bitmap[pos], pos=pos)))


    # Solve the graph with the new node inserted
    to_clear, to_flag = second_order_msm_reduction(MSMG, minecount_first=True, verbose=verbose)

    return MSMG, to_clear, to_flag

def find_num_solutions(MSMG:MSM_Graph, min_n=None, max_n=None, seed=None, verbose=0):
    """
    Find solutions using branch and bound. Faster if connected component.
    """
    counts = dict()
    flat_graph = MSMG.flatten()

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
    
    # TODO, I think there's a way to use connected components before branching and bounding.

    # Otherwise, do branch and bound
    # Select arbitrary coordinate covered by one MSM
    coord = select_branch_coord(flat_graph, seed)
    bitmap = Bitmap(*flat_graph[0].bitmap().shape)
    bitmap[coord] = 1

    # Count the cases if a mine is present at the selected coordinate
    if verbose >= 3: print(f'Try mine at {coord}')
    MSMG_mine, to_clear_m, to_flag_m = try_step_msm(MSMG, mine_bitmap=bitmap, verbose=verbose)
    num_mines = len(to_flag_m)
    num_with_mine = find_num_solutions(MSMG_mine, min_n=min_n, max_n=max_n, seed=seed, verbose=verbose)
    for n, count in num_with_mine.items():
        if n+num_mines not in counts: counts[n+num_mines] = 0
        counts[n+num_mines] += count
    
    # Count the cases if a mine is not present at the selected coordinate
    if verbose >= 3: print(f'Try clear at {coord}')
    MSMG_clear, to_clear_c, to_flag_c = try_step_msm(MSMG, clear_bitmap=bitmap, verbose=verbose)
    num_mines = len(to_flag_c)
    num_wo_mine = find_num_solutions(MSMG_clear, min_n=min_n, max_n=max_n, seed=seed, verbose=verbose)
    for n, count in num_wo_mine.items():
        if n+num_mines not in counts: counts[n+num_mines] = 0
        counts[n+num_mines] += count

    return counts
    
def find_solutions(MSMG:MSM_Graph, stats:SolverStats=None, seed=None, verbose=0) -> SolutionSet:
    """
    Find solutions using branch and bound. Faster if connected component.
    Requires all nodes in MSMG to have the number of mines defined
    """
    flat_graph = MSMG.flatten()

    if len(flat_graph) == 0:
        return SolutionSet() # empty solution set
    
    # If one MSM, the solutions are just the combinations
    if len(flat_graph) == 1:
        bitmap = flat_graph[0].bitmap()
        n = flat_graph[0].n()
        solns = SolutionSet(bitmap)
        assert flat_graph[0].n() is not None
        for combo in combinations(bitmap.decimate(), n):
            solns.add_solution(reduce(lambda b1, b2: b1+b2, combo))
        return solns
    
    # Otherwise, do branch and bound
    components = seperate_connected_msm_components(MSMG)
    solutions = SolutionSet()
    for component in components:
        # Select arbitrary coordinate covered by one MSM
        flat_graph = component.flatten()
        coord = select_branch_coord(flat_graph, seed)
        bitmap = Bitmap(*flat_graph[0].bitmap().shape)
        bitmap[coord] = 1

        # Count the cases if a mine is present at the selected coordinate
        if verbose >= 3: print(f'Try mine at {coord}')
        try:
            MSMG_mine, to_clear_m, to_flag_m = try_step_msm(component, mine_bitmap=bitmap, verbose=verbose)
            soln_with_mine = find_solutions(MSMG_mine, stats=stats, seed=seed, verbose=verbose)
            to_flag_m.add(coord)
            soln_with_mine.expand_solutions(component.bitmap(), to_flag_m)
            if verbose >= 4: print(f'Solutions with mine at {coord}:\n{soln_with_mine}')
        except AssertionError:
            soln_with_mine = SolutionSet()
            if stats is not None: stats.add_uncaught_logic()
            if verbose >= 2: print(f'Uncaught logic pattern found if mine at {coord}\a')
        
        # Count the cases if a mine is not present at the selected coordinate
        try:
            if verbose >= 3: print(f'Try clear at {coord}')
            MSMG_clear, to_clear_c, to_flag_c = try_step_msm(component, clear_bitmap=bitmap, verbose=verbose)
            soln_wo_mine = find_solutions(MSMG_clear, stats=stats, seed=seed, verbose=verbose)
            soln_wo_mine.expand_solutions(component.bitmap(), to_flag_c)
            if verbose >= 4: print(f'Solutions without mine at {coord}:\n{soln_wo_mine}')
        except AssertionError:
            soln_wo_mine = SolutionSet()
            if stats is not None: stats.add_uncaught_logic()
            if verbose >= 2: print(f'Uncaught logic pattern found if clear at {coord}\a')

        component_solutions = SolutionSet.merge_solution_sets(soln_with_mine, soln_wo_mine)
        if verbose >= 4: print(f'Solutions for component {component}:\n{component_solutions}')

        solutions = SolutionSet.combine_solution_sets(solutions, component_solutions)

    return solutions

def select_branch_coord(flat_graph, seed=None, mode='mid'):
    if seed: torch.manual_seed(seed)
    match mode:
        case 'first':
            random_bitmap = flat_graph[0].bitmap()
            random_candidates = random_bitmap.nonzero()
            coord = random_candidates[0]
            return coord
        case 'mid':
            random_bitmap = flat_graph[len(flat_graph)//2].bitmap()
            random_candidates = random_bitmap.nonzero()
            coord = random_candidates[len(random_candidates)//2]
            return coord
        case 'random':
            random_bitmap = flat_graph[torch.randint(len(flat_graph),(1,)).item()].bitmap()
            random_candidates = random_bitmap.nonzero()
            coord = random_candidates[torch.randint(len(random_candidates),(1,))]
            return coord