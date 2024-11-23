from analysis.msm_builder import get_msm_graph, CKEY
from analysis.msm_graph import MSM_Graph
from analysis.msm_analysis import find_solutions, seperate_connected_msm_components, reduce_and_seperate_msm_graph
from analysis.solution_set import SolutionSet
from analysis.utils import merge_disjoint_counts
from core.board import Board
from core.bitmap import Bitmap
from solver.solver_stats import SolverStats
import torch
from math import comb

def calculate_probabilities(B: Board, stats:SolverStats=None, verbose=0):
    """
    Takes a board and computes probability for each square using exhaustive count
    """
    num_mines = B.n
    B, msm, _, _ = get_msm_graph(B, verbose=verbose)
    num_mines -= B.M.sum()

    # Seperate into connected components for speed (since counts of connected components are independent)
    components = seperate_connected_msm_components(msm)
    solutions:list[SolutionSet] = []
    counts = []
    # Find the counts for each connected component
    for component in components:
        if CKEY in component and len(component[CKEY]) > 0: 
            # If complement msm, ignore count (since don't know number of mines yet)
            solutions.append(None)
            counts.append(None)
        else:
            solutions.append(find_solutions(component, stats=stats, verbose=verbose))
            counts.append(solutions[-1].get_solution_counts())
            if verbose == 3: print(f'number of solutions for component\n{component.bitmap()}: {counts[-1]}')
            if verbose >= 4: print(f'solutions for component\n{component.bitmap()}: {solutions[-1]}')

    probabilities = compute_probability_from_solution_counts(components, solutions, counts, B.remaining_mines(), verbose=verbose)

    # If cell was marked to be flagged, its probability is 1
    for coord in B.get_mines():
        probabilities[coord] = 1
    
    assert abs(torch.sum(probabilities)-B.n) < 1e-3, f'Error in probability calculation {torch.sum(probabilities)-B.n}\n{probabilities}'
    return probabilities


def compute_possibilities(counts, remaining_mines, complement_size):
    # merge counts into a larger pool for computing total number of continuations (faster than merging solutions)
    total_counts = {}
    for count in counts:
        if count is None: continue
        total_counts = merge_disjoint_counts(total_counts, count)

    total_possibilities = 0
    complement_possibilities = 0
    if len(total_counts) == 0:
        total_possibilities = comb(complement_size, remaining_mines)
        complement_possibilities = comb(complement_size - 1, remaining_mines - 1)
    # Iterate over counts to find the total number of valid continuations
    for used_mines, count in total_counts.items():
        if remaining_mines < used_mines: continue # Skip if count requires more mines than what's available
        if used_mines + complement_size < remaining_mines: continue # Skip if not enough mines to solve
        if complement_size == 0:
            # If there are no complement items, just add the count
            total_possibilities += count
            continue
        total_possibilities += count * comb(complement_size, remaining_mines - used_mines)
        # If there are mines remaining for the complement space, count the number of solutions for complement space
        if remaining_mines - 1 < used_mines or complement_size <= 0: continue
        complement_possibilities += count * comb(complement_size - 1, remaining_mines - used_mines - 1)

    return total_counts, total_possibilities, complement_possibilities


def compute_probability_from_solution_counts(
        components:list[MSM_Graph], # List of disjoint solvable regions in the board TODO this isn't really needed as solutions should contain a bitmap
        solutions:list[SolutionSet], # List of solutions corresponding to the components
        counts:list[dict[int,int]], # List of the number of solutions for each corresponding solution
        remaining_mines, # number of remaining mines
        return_possibilities=False,
        verbose=0
        ):
    
    c_size = [c[CKEY][0].size() for c in components if CKEY in c][0] if any(CKEY in c and len(c[CKEY])>0 for c in components) else 0
    total_counts, total_possibilities, c_possibilities = compute_possibilities(counts, remaining_mines, c_size)
    if verbose >= 3: print(f'number of total solutions: {total_counts}')
    if verbose >= 3: print(f'number of mines to place: {remaining_mines}')
    if verbose >= 3: print(f'total possibilities: {total_possibilities}')
    c_prob = c_possibilities / total_possibilities # probability of mine in complement region

    probabilities = torch.zeros(*components[0].bitmap().shape)
    for i, (component, solution) in enumerate(zip(components, solutions)):
        # Iterate over the 1 bits in the merged MSM (i.e all the possible locations for the count group)
        coords = component.bitmap().nonzero()
        for coord in coords:
            #Skip complement set
            if solution is None:
                probabilities[coord] = c_prob
                continue
            
            # Insert a mine at square and reduce board
            if verbose >= 3: print(f'computing probability at {coord}')
            # Find number of solutions with mine present
            cond_counts = solution.get_solution_counts_with_coords(mine_coords=[coord])
            if verbose >= 3: print(f'if mine at {coord}, component counts: {cond_counts}')
            # Skip computation if no solutions have mine at location (to avoid synthesis of solutions via combination)
            if len(cond_counts) == 0: 
                probabilities[coord] = 0
                continue
            # Merge counts of other regions
            for j, count in enumerate(counts):
                if i == j or count is None: continue
                cond_counts = merge_disjoint_counts(cond_counts, count)
            if verbose >= 3: print(f'number of non-complement solutions given mine at {coord}: {cond_counts}')

            # Find the number of solutions with the mine given the total mine count
            possibilities = 0
            for n, count in cond_counts.items():
                if remaining_mines < n: continue
                if n + c_size < remaining_mines: continue
                possibilities += count * max(1, comb(c_size, remaining_mines - n))
            if verbose >= 3: print(f'possibilities if mine at {coord}: {possibilities}')
            # Divide to get the probability
            probabilities[coord] = possibilities / total_possibilities

    if return_possibilities:
        return probabilities, total_possibilities
    return probabilities


# ===== Safety Functions =====

def calculate_safety(B: Board, order=1, prune_threshold=0.8, brute_threshold=1000, early_stop=True, stats:SolverStats=None, verbose=0, return_imm=False):
    # base case
    if order == 0:
        return torch.ones(B.shape)


def _safety_helper(
        B:Board,
        order,
        components:list[MSM_Graph], 
        solutions:list[SolutionSet], 
        progress_coords,
        prune_threshold=0.8,
        stats:SolverStats=None,
        verbose=0
        ):
    if order <= 0:
        return torch.ones(B.shape)

    if B.unknown().sum() - B.remaining_mines() - len(progress_coords) == 0:
        return torch.ones(B.shape)
    
    # TODO if progress_coords is nonempty, skip computation and immediately recurse
    if len(progress_coords) > 0:
        coord = progress_coords.pop()
        return _safety_at_coord_helper()

    # Get components, solutions, and counts, recomputing if needs to 
    components_t, solutions_t, counts_t = [], [], []
    pruned = []
    for component, solution in zip(components, solutions):
        comps = seperate_connected_msm_components(component)
        for comp in comps:
            components_t.append(comp)
            if CKEY in comp:
                solutions_t.append(None)
                counts_t.append(None)
                pruned = (comp[CKEY].bitmap().neighbor_count_map().bitmap > 3).nonzero() # TODO need a more elegant interface in bitmap to avoid direct access
            else:
                if solution is None:
                    solutions_t.append(find_solutions(comp, stats=stats, verbose=verbose))
                elif solution.bitmap < comp.bitmap():
                    solutions_t.append(solution.clone().shrink_bitmap(comp.bitmap()))
                else:
                    solutions_t.append(solution.clone())
                counts_t.append(solutions_t[-1].get_solution_counts())
    components, solutions, counts = components_t, solutions_t, counts_t

    immediate_probability, total_possibilities = compute_probability_from_solution_counts(components, solutions, counts, B.remaining_mines(), return_possibilities=True, verbose=verbose)
    immediate_safety = 1 - immediate_probability

    # TODO, Pruning strategy
    max_safety = max([immediate_safety[coord] for coord in B.unknown().nonzero() if coord not in pruned])
    if max_safety == 1:
        progress_coords = [coord for coord in B.unknown().nonzero() if immediate_safety[coord]==1]
        
    safety = torch.zeros(B.shape)
    for coord in B.unknwon().nonzero():
        if coord in pruned or immediate_safety[coord] < prune_threshold * max_safety: continue
        safety[coord] = _safety_at_coord_helper(B, order, coord, components, solutions, total_possibilities, prune_threshold, stats, verbose)

    return safety

def _safety_at_coord_helper(
        B:Board, 
        order,
        coord, 
        components:list[MSM_Graph], 
        solutions:list[SolutionSet], 
        total_possibilities,
        prune_threshold=0.8,
        stats=None,
        verbose=0
        ):
    # Figure out which solution sets get affected by the existance of a number at coord
    c_size = [c[CKEY][0].size() for c in components if CKEY in c][0] if any(CKEY in c and len(c[CKEY])>0 for c in components) else 0
    coord_neigh = B.neighbor_mask(*coord)
    affected_comps = []
    curr_comp_ind = None
    new_region = Bitmap(*B.shape)
    for i, component in enumerate(components):
        overlap = component.bitmap() * coord_neigh
        if overlap.any():
            if CKEY in component and len(component[CKEY]) > 0: 
                new_region = overlap
            elif component.bitmap()[coord] == 0:
                affected_comps.append(i)
        if component.bitmap()[coord] == 1: curr_comp_ind = i
    
    # Create a larger solution set including the neighborhood
    #coord_in_complement = CKEY in components[curr_comp_ind] and len(components[curr_comp_ind][CKEY]) > 0
    #if coord_in_complement: 
    #    curr_solns = SolutionSet.powerset(new_region)
    #else:
    #    curr_solns:SolutionSet = solutions[curr_comp_ind].clone()

    #if new_region.sum() != 0:
    #    curr_solns.add_region(new_region)
        
    # Recreate the graph for the affected components
    curr_component = MSM_Graph.from_single(Bitmap.coords(*B.shape, [coord]), 0, coord)
    components_aug = []
    solutions_aug = [] # [SolutionSet.powerset(new_region)]
    for i in range(len(components)):
        if CKEY in components[i]:
            complement = components[i].bitmap() - new_region
            if complement.sum() > 0:
                components_aug.append(MSM_Graph.from_single(complement, pos=CKEY))
                solutions_aug.append(None)
        elif i in affected_comps:
            for node in components[i].clone():
                curr_component.new_node(node.bitmap, node.n, node.pos)
            #solutions_aug[0] = SolutionSet.combine_solution_sets(solutions_aug[0], solutions[i])
        else:
            components_aug.append(components[i].clone())
            solutions_aug.append(solutions[i].clone())

    # Recompute the solutions for the new component
    for component in reduce_and_seperate_msm_graph(curr_component):
        pass

    # Compute solution counts for each number
    #num_counts = solutions_aug[0].get_solution_counts_for_numbers(coord, clear_coords=[coord])
    if verbose >= 4: print(f"Given {coord} clear, {num_counts}: {solutions_aug[0]}")

    # Compute recursive safety for each possible number
    safety = 0
    for num, cond_counts in num_counts.items():
        if verbose >= 3: print(f"Trying {num} at {coord} with counts {cond_counts}")

        # TODO add number MSM, step with 2nd order reducction, split components and solutions if necessary, compute probability, recurse

        # merge counts
        for i, soln in enumerate(solutions):
            count = soln.get_solution_counts()
            if i == curr_comp_ind or i in affected_comps or count is None: continue
            cond_counts = merge_disjoint_counts(cond_counts, count)
        
        possibilities = 0
        for n, count in cond_counts.items():
            if B.remaining_mines() < n: continue
            if n + c_size - new_region.sum() < B.remaining_mines(): continue
            possibilities += count * max(1, comb(c_size - new_region.sum() - (1 if coord_in_complement else 0), B.remaining_mines() - n))
        if possibilities == 0: continue
        # Divide to get the probability of the number
        num_prob = possibilities / total_possibilities

        B_aug = B.reduce()
        B_aug.set_clear(*coord, num)
        # TODO add trivial first order msm induced by number, and apply second order logic to components

        recursive_safety = _safety_helper(B_aug, 
                                          order-1, 

                                          prune_threshold=prune_threshold, 
                                          stats=stats, 
                                          verbose=verbose)
        safety += num_prob * torch.max(recursive_safety)
    
    return safety