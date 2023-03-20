from core.bitmap import Bitmap

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

class SolutionSet:
    def __init__(self, bitmap:Bitmap=None):
        self.bitmap = bitmap
        self.solutions = set()
        self.coords = set() if self.bitmap is None else bitmap.nonzero()
        self.coord_solutions = {coord: set() for coord in self.coords}
    def __repr__(self) -> str:
        return '{'+',\n'.join([repr(soln) for soln in self.solutions])+'}'
    def add_solution(self, b:Bitmap):
        soln = Solution(b)
        self.solutions.add(soln)
        for coord in b.nonzero():
            self.coord_solutions[coord].add(soln)
    def get_solutions_with_mines(self, mine_coords):
        if not all([coord in self.coords for coord in mine_coords]): return set()
        return set.intersection(*[self.coord_solutions[coord] for coord in mine_coords])
    def expand_solutions(self, new_bitmap, mine_coords):
        self.bitmap = new_bitmap
        # TODO
        pass
    @staticmethod
    def merge_solution_sets(ss1, ss2):
        # TODO
        pass
    