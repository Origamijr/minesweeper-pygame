from core.board import Board
import time

class SolverStats:
    def __init__(self, board: Board, player=True):
        self.board = board
        self.player = player
        self.clicks = 0
        self.uncaught_logic = 0
        self.guesses = 0
        self.risk = 0
        self.steps = []
        self.bbbv = None
        self.time = 0
        self._timer = None
    
    def __repr__(self):
        if self.player:
            return f'[Time: {self.time:.3f}, 3BV/s: {self.get_3bvs():.3f}, Clicks: {self.clicks}, Efficiency: {self.get_efficiency()}, Risk: {self.risk*10:.2f}%]'
        else:
            return f'[Time: {self.time:.3f}, Efficiency: {int(self.get_efficiency()*100)}%, Risk: {self.risk*100:.2f}%, Guesses: {self.guesses}, Uncaught Logic: {self.uncaught_logic}]'

    def start_timer(self):
        self._timer = time.perf_counter()

    def end_timer(self):
        self.time += time.perf_counter() - self._timer
        self._timer = None

    def add_click(self):
        self.clicks += 1

    def add_step(self, coord):
        self.steps.append((time.perf_counter() - self._timer, coord))

    def add_uncaught_logic(self):
        self.uncaught_logic += 1

    def add_guess(self):
        self.guesses += 1

    def apply_risk(self, risk):
        self.risk = 1 - (1 - self.risk) * (1 - risk)

    def get_3bv(self):
        return self.board.get_3bv()
    
    def get_3bvs(self):
        assert self.time > 0
        return self.board.get_3bv() / self.time

    def get_efficiency(self):
        return self.board.get_3bv() / self.clicks