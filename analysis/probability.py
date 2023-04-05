from analysis.msm_builder import get_msm_graph, CKEY
from analysis.msm_analysis import find_solutions, seperate_connected_msm_components, try_step_msm
from analysis.solution_set import SolutionSet
from core.board import Board
from core.bitmap import Bitmap
from solver.solver_stats import SolverStats
import torch
from math import comb
import itertools

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
    bitmaps:list[Bitmap] = []
    counts = []
    c_size = 0
    total_counts = dict()
    # Find the counts for each connected component
    for component in components:
        bitmaps.append(component.bitmap())
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

            # merge counts into a larger pool for computing total number of continuations
            total_counts = __merge_disjoint_counts(total_counts, counts[-1])

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
        if num_mines - 1 < n or c_size == 0: continue
        c_possiblities += count * comb(c_size - 1, num_mines - n - 1)
    if verbose >= 3: print(f'total possibilities: {total_possibilities}')
    c_prob = c_possiblities / total_possibilities # probability of mine in complement region

    # Compute probability for each unknown square
    probabilities = torch.zeros(B.rows, B.cols)
    for i, (bitmap, component, solution) in enumerate(zip(bitmaps, components, solutions)):
        # Iterate over the 1 bits in the merged MSM (i.e all the possible locations for the count group)
        coords = bitmap.nonzero()
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
                cond_counts = __merge_disjoint_counts(cond_counts, count)
            if verbose >= 3: print(f'number of non-complement solutions given mine at {coord}: {cond_counts}')

            # Find the number of solutions with the mine given the total mine count
            possibilities = 0
            for n, count in cond_counts.items():
                if num_mines < n: continue
                if n + c_size < num_mines: continue
                possibilities += count * max(1, comb(c_size, num_mines - n))
            if verbose >= 3: print(f'possibilities if mine at {coord}: {possibilities}')
            # Divide to get the probability
            probabilities[coord] = possibilities / total_possibilities

    # If cell was marked to be flagged, its probability is 1
    for coord in B.get_mines():
        probabilities[coord] = 1
    
    assert abs(torch.sum(probabilities)-B.n) < 1e-3, f'Error in probability calculation {torch.sum(probabilities)-B.n}\n{probabilities}'
    return probabilities

def __merge_disjoint_counts(counts1, counts2):
    # Merge counts of dijoint sets
    if len(counts1) == 0: return counts2
    if len(counts2) == 0: return counts1
    counts = dict()
    for (n1, count1), (n2, count2) in itertools.product(counts1.items(), counts2.items()):
        if not n1+n2 in counts: counts[n1+n2] = 0
        counts[n1+n2] += count1*count2
    return counts