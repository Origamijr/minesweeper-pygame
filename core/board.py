import torch
import torch.nn.functional as F
import copy
import itertools
import numpy as np
from core.bitmap import Bitmap

NEIGHBOR_KERNEL = torch.tensor([[[[1.,1.,1.],
                                  [1.,0.,1.],
                                  [1.,1.,1.]]]])

class Board:
    def __init__(self, rows, cols, n, M:Bitmap=None, C:Bitmap=None, N:Bitmap=None):
        self.rows, self.cols, self.n = rows, cols, n

        # Knowledge sets
        self.M = M if M is not None else Bitmap(rows, cols)
        self.C = C if C is not None else Bitmap(rows, cols)
        self.N = N if N is not None else Bitmap.neg_ones(rows, cols) # -1 indicates don't care

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def empty(rows, cols):
        return Board(rows, cols, 0, C=Bitmap.ones(rows, cols), N=Bitmap(rows, cols))
    
    @staticmethod
    def random_complete(rows, cols, n, seed=None, exclude=[]):
        assert n < rows * cols
        B = Board.empty(rows, cols)
        coords = set(itertools.product(range(rows), range(cols)))
        for e in exclude: coords.remove(e)
        np.random.seed(seed)
        ind = list(np.random.choice(np.arange(len(coords)), n, replace=False))
        for x,y in np.array(list(coords))[ind]:
            B.add_mine(x, y)
        return B

    def is_valid(self, ignore_count=False):
        # Check mutually exclusive information
        if (self.M * self.C).any(): return False

        if ignore_count: return True
        # Check valid number of mines
        if self.M.sum() > self.n: return False

        # Check N is consistent
        sum_K_M = self.M.neighbor_count_map()
        sum_K_C = self.C.neighbor_count_map()
        mask_N = self.number_mask()
        return not ((sum_K_M > self.N) * mask_N).any() or ((self.N > 8 - sum_K_C) * mask_N).any()
    
    def is_complete(self):
        return self.is_valid() and (self.M + self.C).all()

    class BoardCell:
        # Wrapper class to return a single slice of the board
        def __init__(self, M, C, N):
            self.M, self.C, self.N = M.clone(), C.clone(), N.clone()
        def __repr__(self):
            return f"[{self.M},\n{self.C},\n{self.N}]"

    def __getitem__(self, key):
        return self.BoardCell(self.M[key], self.C[key], self.N[key])
    
    def __gt__(self, other):
        # Override > so B_1 > B_2 means B_2 is a valid continuation of B_1
        if self.n != other.n: return False
        if (self.M > other.M).any(): return False
        if (self.C > other.C).any(): return False
        mask_N = self.number_mask()
        return mask_N*self.C*self.N == mask_N*self.C*other.N
        
    def __eq__(self, other):
        # Override = so B_1 == B_2 means the unknown squares have the same probabilities
        # TODO: consider allowing if square has probability 1 of being a mine, it's still equivalent to a mine

        # Reduce each board
        reduced_1 = self.reduce()
        reduced_2 = other.reduce()
        
        return (reduced_1.M == reduced_2.M) and (reduced_1.C == reduced_2.C) and (reduced_1.N == reduced_2.N)
    
    def __repr__(self):
        assert self.is_valid(ignore_count=True), "Invalid Board"
        s = '-' * (self.cols + 2) + f'{self.n-self.M.sum()}/{self.n}\n'
        for x in range(self.rows):
            s += '|'
            for y in range(self.cols):
                if self.C[x,y] == 1:
                    if self.N[x,y] != -1:
                        s += f'{int(self.N[x,y])}'
                    else:
                        s += 'X'
                elif self.M[x,y] == 1:
                    s += '*'
                else:
                    s += ' '
            s += '|\n'
        s += '-' * (self.cols + 2)
        return s

    def add_mine(self, x, y):
        if self.M[x,y] == 0:
            self.n += 1
            self.M[x,y] = 1
            self.C[x,y] = 0
            self.N[x,y] = -1

            # Update neighboring N
            N_inc = self.neighbor_mask(x, y)
            mask_N = self.number_mask()
            self.N = self.N.add(N_inc * mask_N)

    def subtract_mine(self, x, y, compute_n=True):
        if self.M[x,y] == 1:
            assert self.n >= 1
            self.n -= 1
            self.M[x,y] = 0
            self.C[x,y] = 1

            # update neighboring N
            N_inc = self.neighbor_mask(x, y)
            mask_N = self.number_mask()
            self.N = self.N.sub(N_inc * mask_N)
            
            # compute N if complete information of area around cell
            if compute_n and (self.M + self.C) * N_inc == N_inc:
                self.N[x,y] = (self.M * N_inc).sum()

    def unknown(self) -> Bitmap:
        return Bitmap.ones(self.rows, self.cols) - self.M - self.C
    
    def get_3bv(self):
        counted = Bitmap(self.rows, self.cols)
        bbbv = 0
        for zero_coord in self.N.get_mask(0).nonzero():
            # One click for each opening
            if counted[zero_coord] == 1: continue
            for visited_coord in self.get_opening(*zero_coord):
                counted[visited_coord] = 1
            bbbv += 1
        for coord in self.C.nonzero():
            # One click for everything else
            if counted[coord] == 1: continue
            bbbv += 1
        return bbbv

    def neighbor_mask(self, x, y):
        n_mask = Bitmap(self.rows, self.cols)
        n_mask[x,y] = 1
        return n_mask.neighbor_count_map()

    def number_mask(self):
        return -self.N.get_mask(-1)
            
    def set_mine(self, x, y, v=1):
        if self.M[x,y] + self.C[x,y] >= 1: return False
        self.M[x,y] = v
        return True

    def set_clear(self, x, y, N=-1):
        if self.M[x,y] + self.C[x,y] >= 1: return False
        self.C[x,y] = 1
        self.N[x,y] = N
        return True

    def get_neighbor_inds(self, x, y):
        deltas = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        return set([(x+dx, y+dy) for dx,dy in deltas if 0 <= x+dx < self.rows and 0 <= y+dy < self.cols])
    
    def get_opening(self, x, y):
        # Perform a DFS to find all cells in an opening (i.e. all connected 0s and their neighbors)
        stack = [(x,y)]
        opening = set()
        while stack:
            i, j = stack.pop()
            opening.add((i,j))
            if self.N[i,j] != 0: continue
            for ni, nj in self.get_neighbor_inds(i, j):
                if (ni, nj) in opening: continue
                stack.append((ni, nj))
        return opening
    
    def get_mines(self):
        return self.M.nonzero()
    
    def reduce(self):
        B = self.copy()
        for mine_coord in B.get_mines():
            B.subtract_mine(*mine_coord)
        return B
    
    def project_from(self, other):
        assert self > other
        mask = self.C * self.N.get_mask(-1)
        self.N = (self.N.mul(-mask)).add(other.N.mul(mask))