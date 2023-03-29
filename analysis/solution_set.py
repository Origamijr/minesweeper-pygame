import torch
import torch.nn.functional as F
from core.bitmap import Bitmap
import itertools

# TODO Format code nicer (code got longer than I thought)

class Solution:
    # Basically a wrapper to bitmap with a different name
    # Bitmap of locations of mines in a possible solution
    def __init__(self, M:Bitmap):
        self.M = M
        self.n = M.sum()
    def __hash__(self):
        return hash(self.M)
    def __eq__(self, other):
        return self.M == other.M
    def __repr__(self) -> str:
        return repr(self.M)
    def __add__(self, other):
        return Solution(self.M + other.M)
    def __iadd__(self, other):
        self.M += other.M
        self.n = self.M.sum()
        return self
    
class SolutionBitmap:
    # ArrayList implementation of list of solutions with torch tensor as underlying representation
    def __init__(self, board_shape, init_capacity=32):
        self.rows, self.cols = board_shape
        self.curr_element = -1
        self.bitmap = torch.zeros(init_capacity, self.rows * self.cols, dtype=torch.int8)
        self.trimmed = False
    def _coord2col(self, x, y):
        return x * self.cols + y
    def _expand(self, amount):
        self.bitmap = F.pad(self.bitmap, (0,0,0,amount))
    def _trim_rows(self):
        self.bitmap = self.bitmap[:self.curr_element+1,:]
    def _trim_cols(self):
        self.trimmed = True
        self.bitmap = self.bitmap[:,self.bitmap.abs().sum(dim=0).bool().flatten()]
    def capacity(self):
        return self.bitmap.shape[0]
    def add(self, solution:Solution):
        assert not self.trimmed
        self.curr_element += 1
        if self.curr_element >= self.capacity(): self._expand(self.capacity())
        self.bitmap[self.curr_element,:] = solution.M.flatten()
    def add_mine_to_all_solutions(self, mine_coords):
        self.bitmap[:,[self._coord2col(*coord) for coord in mine_coords]] = 1
        self.bitmap[self.curr_element+1:,] = 0
    def concatenate(self, other):
        self._trim_rows(), other._trim_rows()
        new_solution_bitmap = SolutionBitmap((self.rows, self.cols), init_capacity=0)
        new_solution_bitmap.bitmap = torch.cat([self.bitmap, other.bitmap], dim=0)
        new_solution_bitmap.curr_element = new_solution_bitmap.bitmap.shape[0] - 1
        return new_solution_bitmap
    def solution_intersection_mask(self, coords, val=1):
        self._trim_rows() # There might be a way to trim cols too if coords are tracked and indexed instead of computed
        bitmap = self.bitmap if val else 1-self.bitmap
        return torch.prod(bitmap[:,[self._coord2col(*coord) for coord in coords]], dim=1, dtype=torch.int8)
    def get_solution_minecount_with_mines(self, mine_coords):
        row_mask = self.solution_intersection_mask(mine_coords, val=1) > 0
        return torch.sum(self.bitmap[row_mask,:], dim=1, dtype=torch.int)


class SolutionSet:
    def __init__(self, bitmap:Bitmap=None):
        self.bitmap = bitmap
        self.solutions = []
        self.solution_bitmap = None if self.bitmap is None else SolutionBitmap(self.bitmap.shape)
    def __repr__(self) -> str:
        return 'Solutions over\n' + repr(self.bitmap) + '\n{'+',\n'.join([repr(soln) for soln in self.solutions])+'}'
    def add_solution(self, b:Bitmap):
        assert self.bitmap is not None and b * self.bitmap == b
        soln = Solution(b)
        self.solutions.append(soln)
        self.solution_bitmap.add(soln)
    def expand_solutions(self, new_bitmap: Bitmap, mine_coords):
        # Extend existing solutions with information of mines outside the set
        assert self.bitmap is None or (new_bitmap >= self.bitmap).all()
        if self.bitmap is None:
            # If no solutions for self, make new mines the only solution
            self.bitmap = new_bitmap
            new_mines = Bitmap.coords(self.bitmap.rows, self.bitmap.cols, mine_coords)
            self.solution_bitmap = SolutionBitmap(self.bitmap.shape)
            self.add_solution(new_mines)
        else:
            # Otherwise append mines to existing solutions
            self.bitmap = new_bitmap
            new_mines = Solution(Bitmap.coords(self.bitmap.rows, self.bitmap.cols, mine_coords))
            for soln in self.solutions:
                soln += new_mines
            self.solution_bitmap.add_mine_to_all_solutions(mine_coords)
    @staticmethod
    def merge_solution_sets(ss1, ss2):
        # Merge two sets with the same bitmap
        if ss1.bitmap is None: return ss2
        if ss2.bitmap is None: return ss1
        assert ss1.bitmap == ss2.bitmap
        ss = SolutionSet(ss1.bitmap)
        ss.solutions = ss1.solutions + ss2.solutions
        ss.solution_bitmap = ss1.solution_bitmap.concatenate(ss2.solution_bitmap)
        return ss
    @staticmethod
    def combine_solution_sets(ss1, ss2):
        # Combine two sets with disjoint bitmap, slow, but worth the time save
        if ss1.bitmap is None: return ss2
        if ss2.bitmap is None: return ss1
        assert not (ss1.bitmap * ss2.bitmap).any(), repr(ss1.bitmap) + '\n' + repr(ss2.bitmap)
        ss = SolutionSet(ss1.bitmap + ss2.bitmap)
        for soln1, soln2 in itertools.product(ss1.solutions, ss2.solutions):
            ss.add_solution(soln1.M + soln2.M)
        return ss
    def get_solution_counts(self):
        return self.get_solution_counts_with_mines([])
    def get_solution_counts_with_mines(self, mine_coords):
        counts = self.solution_bitmap.get_solution_minecount_with_mines(mine_coords)
        if len(counts) == 0: {0: 0}
        return {num_mines.item(): count.item() for num_mines, count in zip(*torch.unique(counts, return_counts=True))}
    

"""
class SolutionSetBak:
    def __init__(self, bitmap:Bitmap=None):
        self.bitmap = bitmap
        self.solutions = []
        self.coords = set() if self.bitmap is None else set(bitmap.nonzero())
        self.coord_solutions = {coord: [] for coord in self.coords}
    def __repr__(self) -> str:
        return 'Solutions over\n' + repr(self.bitmap) + '\n{'+',\n'.join([repr(soln) for soln in self.solutions])+'}'
    def add_solution(self, b:Bitmap):
        soln = Solution(b)
        self.solutions.append(soln)
        for coord in b.nonzero():
            self.coord_solutions[coord].append(soln)
    def expand_solutions(self, new_bitmap: Bitmap, mine_coords):
        # Extend existing solutions with information of mines outside the set
        #if new_bitmap is None: return
        assert self.bitmap is None or (new_bitmap >= self.bitmap).all()
        if self.bitmap is None:
            # If no solutions for self, make new mines the only solution
            self.bitmap = new_bitmap
            rows, cols = self.bitmap.shape
            new_mines = Bitmap.coords(rows, cols, mine_coords)
            self.coords = set(self.bitmap.nonzero())
            self.coord_solutions = {coord: [] for coord in self.coords}
            self.add_solution(new_mines)
        else:
            # Otherwise append mines to existing solutions
            new_coords = new_bitmap - self.bitmap
            self.bitmap = new_bitmap
            rows, cols = self.bitmap.shape
            new_mines = Solution(Bitmap.coords(rows, cols, mine_coords))
            for soln in self.solutions:
                soln += new_mines
            for coord in new_coords.nonzero():
                self.coords.add(coord)
                self.coord_solutions[coord] = list(self.solutions) if coord in mine_coords else [] # all solutions have mines in new coords
    @staticmethod
    def merge_solution_sets(ss1, ss2):
        # Merge two sets with the same bitmap
        if ss1.bitmap is None: return ss2
        if ss2.bitmap is None: return ss1
        assert ss1.bitmap == ss2.bitmap
        ss = SolutionSet(ss1.bitmap)
        # Note that this is a bit slow. If used data structure with O(1) list joining instead, this will be faster. Don't care for random access runtime
        ss.solutions = ss1.solutions + ss2.solutions
        for coord in ss.coords:
            ss.coord_solutions[coord] = ss1.coord_solutions[coord] + ss2.coord_solutions[coord]
        return ss
    @staticmethod
    def combine_solution_sets(ss1, ss2):
        # Combine two sets with disjoint bitmap
        if ss1.bitmap is None: return ss2
        if ss2.bitmap is None: return ss1
        assert not (ss1.bitmap * ss2.bitmap).any(), repr(ss1.bitmap) + '\n' + repr(ss2.bitmap)
        ss = SolutionSet(ss1.bitmap + ss2.bitmap)
        for soln1, soln2 in itertools.product(ss1.solutions, ss2.solutions):
            ss.add_solution(soln1.M + soln2.M)
        return ss
    def get_solutions(self):
        return self.solutions
    def get_solutions_with_mines(self, mine_coords):
        # Returns list of solution that are a superset of the input mines
        if not all([coord in self.coords for coord in mine_coords]): return set()
        if len(mine_coords) == 1: return self.coord_solutions[mine_coords[0]]
        # List intersection is very slow. avoid if possible.
        solns = set.intersection(*[set(self.coord_solutions[coord]) for coord in mine_coords])
        return solns
    def get_solution_groups_with_mines(self, mine_coords):
        # Returns groups of solutions with mines by number of mines
        count_dict = dict()
        for soln in self.get_solutions_with_mines(mine_coords):
            if soln.n not in count_dict: count_dict[soln.n] = []
            count_dict[soln.n].append(soln)
        return count_dict
    def get_solution_counts(self):
        count_dict = dict()
        for soln in self.solutions:
            if soln.n not in count_dict: count_dict[soln.n] = 0
            count_dict[soln.n] += 1
        return count_dict
    def get_solution_counts_with_mines(self, mine_coords):
        count_dict = dict()
        for soln in self.get_solutions_with_mines(mine_coords):
            if soln.n not in count_dict: count_dict[soln.n] = 0
            count_dict[soln.n] += 1
        if len(count_dict) == 0: count_dict[0] = 0
        return count_dict
"""