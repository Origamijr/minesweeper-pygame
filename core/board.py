import torch
from misc.constants import NEIGHBOR_KERNEL
import torch.nn.functional as F

class Board:
    def __init__(self, rows, cols, n, M=None, C=None, N=None):
        self.rows, self.cols, self.n = rows, cols, n

        # Knowledge sets
        self.M = M if M is not None else torch.zeros((1, 1, self.rows, self.cols))
        self.C = C if C is not None else torch.zeros((1, 1, self.rows, self.cols))
        self.N = N if N is not None else -torch.ones((1, 1, self.rows, self.cols)) # -1 indicates don't care

    @staticmethod
    def empty(rows, cols):
        return Board(rows, cols, 0, C=torch.ones((1, 1, rows, cols)), N=torch.zeros((1, 1, rows, cols)))

    def is_valid(self, ignore_count=False):
        if torch.any(self.M + self.C > 1): return False
        if ignore_count: return True
        if torch.sum(self.M) > self.n: return False
        sum_K_M = F.conv2d(F.pad(self.M, (1,1,1,1), value=0), NEIGHBOR_KERNEL)
        sum_K_C = F.conv2d(F.pad(self.C, (1,1,1,1), value=0), NEIGHBOR_KERNEL)
        mask_N = self.N != -1
        return not torch.any((sum_K_M > self.N) * mask_N) or torch.any((self.N > 8 - sum_K_C) * mask_N)
    
    def is_complete(self):
        return torch.all(self.M + self.C == 1).item()

    class BoardCell:
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
        return not torch.any(mask_N*(self.C*self.N == self.C*other.N))
        
    def __eq__(self, other):
        # Override = so B_1 = B_2 means the unknown squares have the same probabilities
        
        # Find areas where one board has mines and the other doesn't
        mine_diff_1 = torch.max(self.M - other.M, 0)
        mine_diff_2 = torch.max(other.M - self.M, 0)

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
        s = '-' * (self.cols + 2) + '\n'
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
            self.M[...,x,y] = 1
            self.C[...,x,y] = 0
            self.N[...,x,y] = -1
            N_inc = torch.zeros((1, 1, self.rows, self.cols))
            N_inc[...,x,y] = 1
            N_inc = F.conv_transpose2d(N_inc, NEIGHBOR_KERNEL)[...,1:-1,1:-1]
            mask_N = self.N != -1
            self.N += N_inc * mask_N
            self.n += 1
            
    def set_mine(self, x, y, v=1):
        self.M[...,x,y] = v

    def set_clear(self, x, y, N):
        self.C[...,x,y] = 1
        self.N[...,x,y] = N