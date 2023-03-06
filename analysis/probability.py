from analysis.msm_builder import get_msm_graph, CKEY
from analysis.msm_analysis import find_num_solutions, seperate_connected_msm_components, try_step_msm
from analysis.utils import flatten_msm_dict, get_one_ind
from core.board import Board
import torch
from math import comb
import itertools

def calculate_probabilities(B: Board, verbose=0):
    """
    Takes a board and computes probability for each square using exhaustive count
    """
    num_mines = B.n
    B, msm, to_clear, to_flag = get_msm_graph(B, verbose=verbose)
    num_mines -= int(torch.sum(B.M).item())

    # Seperate into connected components for speed (since counts of connected components are independent)
    components = seperate_connected_msm_components(msm)

    counts = []
    c_size = 0
    bitmaps = []
    total_counts = dict()
    # Find the counts for each connected component
    for component in components:
        bitmap = torch.zeros(1,1,B.rows,B.cols)
        for node in flatten_msm_dict(component):
            bitmap += node.bitmap()
        bitmaps.append(bitmap)
        if CKEY in component: 
            # If complement msm, ignore count (since don't know number of mines yet)
            c_size = int(component[CKEY][0].size())
            counts.append(None)
        else:
            counts.append(find_num_solutions(component, verbose=verbose))
            if verbose >= 3: print(f'number of solutions for component {flatten_msm_dict(component)}: {counts[-1].items()}')

            # merge counts into a larger pool for computing total number of continuations
            total_counts = __merge_disjoint_counts(total_counts, counts[-1])

    if verbose >= 3: print(f'number of total solutions: {total_counts}')
    if verbose >= 3: print(f'number of mines: {num_mines}')
    total_possibilities = 0
    c_possiblities = 0
    # Iterate over counts to find the total number of valid continuations
    for n, count in total_counts.items():
        if c_size == 0:
            total_possibilities += count
            continue
        if n > num_mines: continue # Skip if count requires more mines than what's available
        total_possibilities += count * comb(c_size, num_mines - n)
        # If there are mines remaining for the complement space, count the number of solutions for complement space
        if num_mines - 1 < n or c_size == 0: continue
        c_possiblities += count * comb(c_size - 1, num_mines - n - 1)
    if verbose >= 3: print(f'total possibilities: {total_possibilities}')
    c_prob = c_possiblities / total_possibilities

    # Compute probability for each unknown square
    probabilities = torch.zeros(B.rows, B.cols)
    for i, (bitmap, component) in enumerate(zip(bitmaps, components)):
        # Iterate over the 1 bits in the merged MSM (i.e all the possible locations for the count group)
        coords = get_one_ind(bitmap[0,0,...] != 0)
        for coord in coords:
            #Skip complement set
            if CKEY in component:
                probabilities[coord] = c_prob
                continue
            
            # Insert a mine at square and reduce board
            if verbose >= 3: print(f'computing probability at {coord}')
            mine_bitmap = torch.zeros(bitmap.shape)
            mine_bitmap[0,0,coord[0],coord[1]] = 1
            reduced_component, _, _ = try_step_msm(component, mine_bitmap=mine_bitmap, verbose=verbose)
            # Find number of solutions with mine present
            cond_counts = find_num_solutions(reduced_component, verbose=verbose)
            # Merge counts of other regions
            for j, count in enumerate(counts):
                if i == j or count is None: continue
                cond_counts = __merge_disjoint_counts(cond_counts, count)
            if verbose >= 3: print(f'number of solutions given mine at {coord}: {cond_counts}')

            # Find the number of solutions with the mine given the total mine count
            possibilities = 0
            for n, count in cond_counts.items():
                if num_mines - 1 < n: continue
                possibilities += count * max(1, comb(c_size, num_mines - n - 1))
            if verbose >= 3: print(f'possibilities if mine at {coord}: {possibilities}')
            # Divide to get the probability
            probabilities[coord] = possibilities / total_possibilities

    # If cell was marked to be flagged, its probability is 1
    for coord in to_flag:
        probabilities[coord] = 1
    
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