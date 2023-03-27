from core.bitmap import Bitmap
import itertools

# TODO Format code nicer (code got longer than I thought)

class Solution:
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

class SolutionSet:
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