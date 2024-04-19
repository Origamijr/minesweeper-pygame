from core.board import Board
from analysis.msm_builder import get_msm_graph
from analysis.probability import calculate_probabilities
from analysis.utils import update_board
from solver.solver_stats import SolverStats

def click_square(B_k: Board, B_t: Board, coord, stats:SolverStats=None, overcount=False):
    # set to clear and clear rest of opening
    added = False
    for opening_coord in B_t.get_opening(*coord):
        if B_k.set_clear(*opening_coord): added = True
    B_k.project_from(B_t)
    
    # update stats
    if added or overcount:
        stats.add_step(coord)
        stats.add_click()

def play_logic(B_k: Board, B_t: Board, stats:SolverStats=None, verbose=0):
    while True:
        _, _, to_clear, to_flag = get_msm_graph(B_k, order = 2, verbose=verbose)
        update_board(B_k, [], to_flag) # Add mines without clicking for sake of completeness
        for clear in to_clear: # TODO explore different orderings of to_clear
            click_square(B_k, B_t, clear, stats=stats)
        if verbose >= 2 and len(to_clear) > 0: print(B_k, stats.clicks, len(to_clear))
        if len(to_clear) == 0: break

def guess_min_mine_probability(B_k: Board, stats:SolverStats=None, verbose=0):
    p = calculate_probabilities(B_k, stats=stats, verbose=verbose)
    if verbose >= 2: print(p)
    unknown_coords = B_k.unknown().nonzero()
    min_coord, min_val = unknown_coords[0], 1
    for coord in unknown_coords:
        if p[coord] < min_val:
            min_coord, min_val = coord, p[coord]
    if stats is not None: stats.apply_risk(min_val)
    return min_coord, min_val

def solve(B_k: Board, B_t: Board, guess_fn=guess_min_mine_probability, first_move=None, stats=None, verbose=0):
    stats = SolverStats(B_k, B_t, player=False) if stats is None else stats
    stats.start_timer()
    if first_move is not None: click_square(B_k, B_t, first_move, stats=stats)
    if verbose >= 1: print(B_k)
    won = False
    while not won:
        # play with logic rules
        play_logic(B_k, B_t, verbose=verbose, stats=stats)
        
        # break if game is complete
        if B_k.C == B_t.C:
            won = True
            break

        # make guess if game isn't over
        guess, risk = guess_fn(B_k, stats=stats, verbose=verbose)
        if risk > 0: stats.add_guess()

        if verbose >= 1: print(B_k, stats.clicks)
        if verbose >= 1: print(f"Guess {guess}")

        # if guess is a mine, end
        if B_t[guess].M == 1: break
        
        # otherwise click
        click_square(B_k, B_t, guess, stats=stats)

    stats.end_timer()
    return won, stats