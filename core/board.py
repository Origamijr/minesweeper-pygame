import torch
import torch.nn.functional as F
import copy
import itertools
import numpy as np

NEIGHBOR_KERNEL = torch.tensor([[[[1.,1.,1.],
                                  [1.,0.,1.],
                                  [1.,1.,1.]]]])

class Board:
    def __init__(self, rows, cols, n, M=None, C=None, N=None):
        self.rows, self.cols, self.n = rows, cols, n

        # Knowledge sets
        self.M = M if M is not None else torch.zeros((1, 1, self.rows, self.cols))
        self.C = C if C is not None else torch.zeros((1, 1, self.rows, self.cols))
        self.N = N if N is not None else -torch.ones((1, 1, self.rows, self.cols)) # -1 indicates don't care

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def empty(rows, cols):
        return Board(rows, cols, 0, C=torch.ones((1, 1, rows, cols)), N=torch.zeros((1, 1, rows, cols)))
    
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
        if torch.any(self.M + self.C > 1): return False

        if ignore_count: return True
        # Check valid number of mines
        if torch.sum(self.M) > self.n: return False

        # Check N is consistent
        sum_K_M = F.conv2d(F.pad(self.M, (1,1,1,1), value=0), NEIGHBOR_KERNEL)
        sum_K_C = F.conv2d(F.pad(self.C, (1,1,1,1), value=0), NEIGHBOR_KERNEL)
        mask_N = self.N != -1
        return not torch.any((sum_K_M > self.N) * mask_N) or torch.any((self.N > 8 - sum_K_C) * mask_N)
    
    def is_complete(self):
        return torch.all(self.M + self.C == 1).item()

    class BoardCell:
        # Wrapper class to return a single slice of the board
        def __init__(self, M, C, N):
            self.M, self.C, self.N = M[0,0], C[0,0], N[0,0]
        def __repr__(self):
            return f"[{self.M},\n{self.C},\n{self.N}]"

    def __getitem__(self, key):
        if not isinstance(key, tuple): key = (key,)
        key = (slice(None),slice(None)) + (key)
        return self.BoardCell(self.M[key], self.C[key], self.N[key])
    
    def __gt__(self, other):
        # Override > so B_1 > B_2 means B_2 is a valid continuation of B_1
        if self.n != other.n: return False
        if torch.any(self.M > other.M): return False
        if torch.any(self.C > other.C): return False
        mask_N = self.N != -1
        return torch.all(mask_N*self.C*self.N == mask_N*self.C*other.N)
        
    def __eq__(self, other):
        # Override = so B_1 = B_2 means the unknown squares have the same probabilities
        # TODO: consider allowing if square has probability 1 of being a mine, it's still equivalent to a mine

        # Find areas where one board has mines and the other doesn't
        mine_diff_1 = torch.max(self.M - other.M, 0).values
        mine_diff_2 = torch.max(other.M - self.M, 0).values

        # Erase those mines and update knowlege sets
        reduced_M_1 = self.M - mine_diff_1
        reduced_C_1 = self.C + mine_diff_1
        mine_sub_1 = F.conv_transpose2d(mine_diff_1, NEIGHBOR_KERNEL)[...,1:-1,1:-1]
        reduced_N_1 = self.N - mine_sub_1
        
        reduced_M_2 = self.M - mine_diff_2
        reduced_C_2 = self.C + mine_diff_2
        mine_sub_2 = F.conv_transpose2d(mine_diff_2, NEIGHBOR_KERNEL)[...,1:-1,1:-1]
        reduced_N_2 = self.N - mine_sub_2
        
        return torch.all(reduced_M_1 == reduced_M_2) \
            and torch.all(reduced_C_1 == reduced_C_2) \
            and torch.all(reduced_N_1 == reduced_N_2)
    
    def __repr__(self):
        assert self.is_valid(ignore_count=True), "Invalid Board"
        s = '-' * (self.cols + 2) + f'{self.n}\n'
        for x in range(self.rows):
            s += '|'
            for y in range(self.cols):
                if self.C[...,x,y] == 1:
                    if self.N[...,x,y] != -1:
                        s += f'{int(self.N[:,:,x,y])}'
                    else:
                        s += 'X'
                elif self.M[...,x,y] == 1:
                    s += '*'
                else:
                    s += ' '
            s += '|\n'
        s += '-' * (self.cols + 2)
        return s

    def add_mine(self, x, y):
        if self.M[...,x,y] == 0:
            self.n += 1
            self.M[...,x,y] = 1
            self.C[...,x,y] = 0
            self.N[...,x,y] = -1

            # Update neighboring N
            N_inc = self.neighbor_mask(x, y)
            mask_N = self.N != -1
            self.N += N_inc * mask_N

    def subtract_mine(self, x, y, compute_n=True):
        if self.M[...,x,y] == 1:
            assert self.n >= 1
            self.n -= 1
            self.M[...,x,y] = 0
            self.C[...,x,y] = 1

            # update neighboring N
            N_inc = self.neighbor_mask(x, y)
            mask_N = self.N != -1
            self.N -= N_inc * mask_N
            
            # compute N if complete information of area around cell
            if compute_n and torch.all((self.M + self.C) * N_inc == N_inc):
                self.N[...,x,y] = torch.sum(self.M * N_inc)

    def neighbor_mask(self, x, y):
        n_mask = torch.zeros((1, 1, self.rows, self.cols))
        n_mask[...,x,y] = 1
        n_mask = F.conv_transpose2d(n_mask, NEIGHBOR_KERNEL)[...,1:-1,1:-1]
        return n_mask

            
    def set_mine(self, x, y, v=1):
        if self.M[...,x,y] + self.C[...,x,y] >= 1: return
        self.M[...,x,y] = v

    def set_clear(self, x, y, N=-1):
        if self.M[...,x,y] + self.C[...,x,y] >= 1: return
        self.C[...,x,y] = 1
        self.N[...,x,y] = N

    def get_neighbors(self, x, y):
        deltas = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        return set([(x+dx, y+dy) for dx,dy in deltas if 0 <= x+dx < self.rows and 0 <= y+dy < self.cols])
    
    def get_opening(self, x, y):
        # Perform a DFS to find all cells in an opening (i.e. all connected 0s and their neighbors)
        stack = [(x,y)]
        opening = set()
        while stack:
            i, j = stack.pop()
            opening.add((i,j))
            if self.N[...,i,j] != 0: continue
            for ni, nj in self.get_neighbors(i, j):
                if (ni, nj) in opening: continue
                stack.append((ni, nj))
        return opening
    
    def get_mines(self):
        return set([tuple(coord.numpy()) for coord in (self.M == 1).nonzero()[:,-2:]])
    
    def reduce(self):
        B = self.copy()
        for mine_coord in B.get_mines():
            B.subtract_mine(*mine_coord)
        return B
    
    def project_from(self, other):
        assert self > other
        mask = (self.C == 1) * (self.N == -1)
        self.N = self.N * torch.logical_not(mask) + other.N * mask