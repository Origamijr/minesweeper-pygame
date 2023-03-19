from core.bitmap import Bitmap

class Solution:
    def __init__(self, M:Bitmap):
        self.M = M
        self.n = M.sum()
    def __hash__(self):
        return hash(self.M)
    def __eq__(self, other):
        return self.M == other.M

class SolutionSet:
    def __init__(self, bitmap:Bitmap):
        self.bitmap = bitmap
        self.coords = bitmap.nonzero()
        self.solutions = set()
    