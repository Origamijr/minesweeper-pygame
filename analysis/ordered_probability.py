from analysis.msm_builder import get_msm_graph, CKEY
from analysis.msm_analysis import find_solutions, seperate_connected_msm_components, try_step_msm
from analysis.solution_set import SolutionSet
from analysis.utils import merge_disjoint_counts
from core.board import Board
from core.bitmap import Bitmap
from solver.solver_stats import SolverStats
import torch
from math import comb
import itertools

def calculate_safety(B: Board, order=1, prune_threshold=0.8, brute_threshold=1000, early_stop=True, stats:SolverStats=None, verbose=0, return_imm=False):
    # base case
    if order == 0:
        return torch.ones(B.shape)

    # Apply inferrence to a board to reduce search space
    num_mines = B.n
    B, msm, _, _ = get_msm_graph(B, verbose=verbose)
    num_mines -= B.M.sum()

    # Break if complete
    if B.is_complete(): return torch.ones(B.shape)

    # Seperate into connected components for speed (since counts of connected components are independent)
    components = seperate_connected_msm_components(msm)
    solutions:list[SolutionSet] = []
    counts = []
    c_size = 0
    total_counts = dict()
    # Find the counts for each connected component
    for component in components:
        if CKEY in component and len(component[CKEY]) > 0: 
            # If complement msm, ignore count (since don't know number of mines yet)
            c_size = int(component[CKEY][0].size())
            solutions.append(None)
            counts.append(None)
        else:
            solutions.append(find_solutions(component, stats=stats, verbose=verbose))
            counts.append(solutions[-1].get_solution_counts())
            if verbose == 3: print(f'number of solutions for component\n{component.bitmap()}: {counts[-1]}')
            if verbose >= 4: print(f'solutions for component\n{component.bitmap()}: {solutions[-1]}')

            # merge counts into a larger pool for computing total number of continuations (faster than merging solutions)
            total_counts = merge_disjoint_counts(total_counts, counts[-1])

    if verbose >= 3: print(f'number of total solutions: {total_counts}')
    if verbose >= 3: print(f'number of mines to place: {num_mines}')
    total_possibilities = 0
    c_possiblities = 0
    if len(total_counts) == 0:
        total_possibilities = comb(c_size, num_mines)
        c_possiblities = comb(c_size - 1, num_mines - 1)
    # Iterate over counts to find the total number of valid continuations
    for n, count in total_counts.items():
        if num_mines < n: continue # Skip if count requires more mines than what's available
        if n + c_size < num_mines: continue # Skip if not enough mines to solve
        if c_size == 0:
            # If there are no complement items, just add the count
            total_possibilities += count
            continue
        total_possibilities += count * comb(c_size, num_mines - n)
        # If there are mines remaining for the complement space, count the number of solutions for complement space
        if num_mines - 1 < n or c_size <= 0: continue
        c_possiblities += count * comb(c_size - 1, num_mines - n - 1)
    if verbose >= 3: print(f'total possibilities: {total_possibilities}')

    # Compute immediate safety for each unknown square
    immediate_safety = torch.zeros(B.rows, B.cols)
    pruned = torch.zeros(B.rows, B.cols)
    for i, (component, solution) in enumerate(zip(components, solutions)):
        # Premerge counts for other components
        merged_counts = dict()
        for j, count in enumerate(counts):
            if i == j or count is None: continue
            merged_counts = merge_disjoint_counts(merged_counts, count)

        # Iterate over the 1 bits in the merged MSM (i.e all the possible locations for the count group)
        coords = component.bitmap().nonzero()
        for coord in coords:
            #Evaluate complement set
            if solution is None:
                immediate_safety[coord] = 1 - (c_possiblities / total_possibilities)
                # Prune elements that aren't effective corners (completely unknown neighbors > 3)
                if brute_threshold is not None and total_possibilities > brute_threshold:
                    if (component.bitmap() * B.neighbor_mask(coord)).sum > 3:
                        pruned[coord] = 1
                continue
            
            # Insert a mine at square and reduce board
            if verbose >= 3: print(f'computing probability at {coord}')
            # Find number of solutions with mine present
            cond_counts = solution.get_solution_counts_with_coords(clear_coords=[coord])
            if verbose >= 3: print(f'if mine at {coord}, component counts: {cond_counts}')
            # Skip computation if no solutions have cleared location
            if len(cond_counts) == 0: 
                immediate_safety[coord] = 0
                continue
            cond_counts = merge_disjoint_counts(cond_counts, merged_counts)
            if verbose >= 3: print(f'number of non-complement solutions given mine at {coord}: {cond_counts}')

            # Find the number of solutions with the mine given the total mine count
            possibilities = 0
            for n, count in cond_counts.items():
                if num_mines < n: continue
                if n + c_size < num_mines: continue
                possibilities += count * max(1, comb(c_size, num_mines - n))
            if verbose >= 3: print(f'possibilities if mine at {coord}: {possibilities}')
            # Divide to get the probability
            immediate_safety[coord] = possibilities / total_possibilities

    not_pruned = (pruned + (immediate_safety < (prune_threshold * immediate_safety.max()))) == 0
    # If only one square isn't pruned, stop early by reutrning the immediate safety
    if not_pruned.sum() == 1:
        return immediate_safety if not return_imm else (immediate_safety, immediate_safety)

    print(immediate_safety)

    # Recursively compute safety for each square that's not pruned
    safety = torch.zeros(B.rows, B.cols)
    for coord in not_pruned.nonzero():
        # Figure out which solution sets get affected by the existance of a number at coord
        coord = tuple(coord.numpy())
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
        coord_in_complement = CKEY in components[curr_comp_ind] and len(components[curr_comp_ind][CKEY]) > 0
        if coord_in_complement: 
            curr_solns = SolutionSet.powerset(new_region)
        else:
            curr_solns:SolutionSet = solutions[curr_comp_ind].clone()

        if new_region.sum() != 0:
            curr_solns.add_region(new_region)
        for comp_ind in affected_comps:
            curr_solns = SolutionSet.combine_solution_sets(curr_solns, solutions[comp_ind])

        # Compute solution counts for each number
        num_counts = curr_solns.get_solution_counts_for_numbers(coord, clear_coords=[coord])

        # Compute recursive safety for each possible number
        for num, cond_counts in num_counts.items():
            # merge counts
            for i, count in enumerate(counts):
                if i == curr_comp_ind or i in affected_comps or count is None: continue
                cond_counts = merge_disjoint_counts(cond_counts, count)
            
            possibilities = 0
            for n, count in cond_counts.items():
                if num_mines < n: continue
                if n + c_size - new_region.sum() < num_mines: continue
                possibilities += count * max(1, comb(c_size - new_region.sum() - (1 if coord_in_complement else 0), num_mines - n))
            if possibilities == 0: continue
            # Divide to get the probability of the number
            num_prob = possibilities / total_possibilities

            B_aug = B.clone()
            B_aug.set_clear(coord[0], coord[1], num)
            recursive_safety = calculate_safety(B_aug, order-1, prune_threshold=prune_threshold, brute_threshold=brute_threshold, stats=stats, verbose=verbose)
            safety[coord] += num_prob * torch.max(recursive_safety)
            
    if return_imm:
        return safety, immediate_safety
    return safety
