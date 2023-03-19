from core.board import Board
from analysis.msm_builder import get_msm_graph
from analysis.probability import calculate_probabilities
from analysis.utils import get_one_ind
import numpy as np
import itertools

def play_logic(B_k: Board, B_t: Board, verbose=0):
    steps = []
    while True:
        B_k, msm, to_clear, _to_flag = get_msm_graph(B_k, order = 2, verbose=verbose)
        for x, y in to_clear:
            from_opening = False
            for dx, dy in itertools.product(range(-1,2), range(-1,2)):
                if 0 <= x+dx < B_t.rows and 0 <= y+dy < B_t.cols:
                    if B_k[x+dx, y+dy].N == 0:
                        from_opening = True
                        break
            if not from_opening: steps.append((x, y))
        B_k.project_from(B_t)
        if verbose >= 2: print(B_k)
        if len(to_clear) == 0: break
    return B_k, steps

def guess_min_mine_probability(B_k: Board, verbose=0):
    p = calculate_probabilities(B_k, verbose=verbose)
    if verbose >= 2: print(p)
    coords = get_one_ind(p != 0)
    min_coord, min_val = coords[0], 1
    for coord in coords:
        if p[coord] < min_val:
            min_coord, min_val = coord, p[coord]
    return min_coord

def solve(B_k: Board, B_t: Board, guess_fn=guess_min_mine_probability, verbose=0):
    steps = []
    while True:
        B_k, ng_steps = play_logic(B_k, B_t, verbose=verbose)
        steps += ng_steps
        if B_k.is_complete(): return True, steps
        steps.append(guess_fn(B_k, verbose=verbose))
        if verbose == 1: print(B_k)
        if verbose >= 1: print(f"Guess {steps[-1]}")
        if B_t[steps[-1]].M == 1: return False, steps
        #if verbose >= 1: print(f"Clear {B_t.get_opening(*steps[-1])}")
        for coord in B_t.get_opening(*steps[-1]):
            B_k.set_clear(*coord)
        B_k.project_from(B_t)