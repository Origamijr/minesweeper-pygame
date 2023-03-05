from analysis.msm_builder import get_msm_graph, CKEY
from analysis.msm_analysis import find_num_solutions, seperate_connected_msm_components, try_step_msm
from analysis.utils import merge_disjoint_counts, flatten_msm_dict, get_one_ind
from core.board import Board
import torch
from math import comb

def calculate_probabilities(B: Board):
    num_mines = B.n
    _, msm, to_clear, to_flag = get_msm_graph(B)

    components = seperate_connected_msm_components(msm)

    counts = []
    c_size = 0
    bitmaps = []
    total_counts = dict()
    for component in components:
        bitmap = torch.zeros(1,1,B.rows,B.cols)
        for node in flatten_msm_dict(component):
            bitmap += node.bitmap()
        bitmaps.append(bitmap)
        if CKEY in component: 
            c_size = int(component[CKEY][0].size())
            counts.append(None)
        else:
            counts.append(find_num_solutions(component))
            total_counts = merge_disjoint_counts(total_counts, counts[-1])

    total_possibilities = 0
    c_possiblities = 0
    for n, count in total_counts.items():
        if n > num_mines: continue
        total_possibilities += count * comb(c_size, num_mines - n)
        if num_mines - n - 1 < 0 or c_size == 0: continue
        c_possiblities += count * comb(c_size - 1, num_mines - n - 1)
    c_prob = c_possiblities / total_possibilities


    probabilities = torch.zeros(B.rows, B.cols)
    for i, (bitmap, component) in enumerate(zip(bitmaps, components)):
        coords = get_one_ind(bitmap[0,0,...] != 0)
        for coord in coords:
            if CKEY in component:
                probabilities[coord] = c_prob
                continue
            mine_bitmap = torch.zeros(bitmap.shape)
            mine_bitmap[0,0,coord[0],coord[1]] = 1
            reduced_component, _, _ = try_step_msm(component, mine_bitmap=mine_bitmap)
            cond_counts = find_num_solutions(reduced_component)
            for j, count in enumerate(counts):
                if i == j or count is None: continue
                cond_counts = merge_disjoint_counts(cond_counts, count)
            possibilities = 0
            for n, count in cond_counts.items():
                possibilities += count * comb(c_size, num_mines - n - 1)
            probabilities[coord] = possibilities / total_possibilities
    for coord in to_flag:
        probabilities[coord] = 1
    
    return probabilities
