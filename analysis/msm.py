from core.board import Board
import torch
import itertools
import copy

def get_msm_graph(B: Board, order=2):
    assert order in [1,2]
    B = B.copy()

    # mine reduce the board completely
    __reduce_board(B)

    class MSM:
        def __init__(self, bitmap, n, size):
            self.bitmap, self.n, self.size = bitmap, n, size
            self.coord = None
        def __hash__(self):
            return hash(self.bitmap)
        def __eq__(self, other):
            return torch.all(self.bitmap == other.bitmap) and self.n == other.n
        def __sub__(self, other):
            d = MSM(self.bitmap - other.bitmap, self.n - other.n, self.size - other.size)
            d.coord = self.coord
            return d
        def __repr__(self):
            return repr(self.bitmap) + '\t' + repr(self.n)
        def set_coord(self, x, y): self.coord = (x,y)

    # Find all first-order MSMs induced by numbered cells
    b_msm = (1 - B.C)[0,0,...] # The trivial MSM created by the area not touching any number
    
    # If minecounting applies, we can mark certain coordinates as probability 1 areas
    to_clear = set()
    to_flag = set()
    
    # Iterate until no more minecounting patterns are found
    minecount = True
    while minecount:
        MSMs = [[set() for c in range(B.cols)] for r in range(B.rows)]

        # Find the MSM induced by each number
        for coord in (B.N > -1).nonzero()[:,-2:]:
            x, y = coord
            
            bitmap = (B.neighbor_mask(*coord) * (1 - B.C))[0,0,...]
            mines = B[x,y].N
            size = torch.sum(bitmap)

            if size == 0: # skip degenerate cases
                B.N[...,x,y] = -1
                continue 

            m = MSM(bitmap, mines, size)
            m.set_coord(x, y)
            MSMs[x][y].add(m)
            
            # Update the trivial msm
            b_msm *= (1 - bitmap)
        
        suggestions = __minecount_reduce(B, __flatten3d(MSMs))
        if suggestions is None: 
            minecount = False
            continue
        to_clear = to_clear.union(suggestions[0])
        to_flag = to_flag.union(suggestions[1])

    # Find all second-order MSMs by partitioning MSMs by subsets
    if order >= 2:
        minecount = True
        while minecount:
            minecount = False
            new_msm = True
            new_MSMs = [[set() for c in range(B.cols)] for r in range(B.rows)]
            curr_MSMs = [[set() for c in range(B.cols)] for r in range(B.rows)]
            __union2dset(curr_MSMs, MSMs)
            last_MSMs = curr_MSMs
            first_iter = True
            while new_msm:
                new_msm = False
                to_remove = [[set() for c in range(B.cols)] for r in range(B.rows)]
                for x, y in itertools.product(range(B.rows), range(B.cols)):
                    if len(MSMs[x][y]) == 0: continue
                    for dx, dy in itertools.product(range(3), range(3)):
                        if (dx, dy) == (0, 0) or x+dx >= B.rows or y+dy >= B.cols: continue
                        for m1, m2 in itertools.product(last_MSMs[x][y], curr_MSMs[x+dx][y+dy]):
                            if m1 == m2: continue
                            intersection = m1.bitmap * m2.bitmap
                            if torch.all(intersection == m1.bitmap):
                                to_remove[x+dx][y+dy].add(m2)
                                new_MSMs[x+dx][y+dy].add(m2 - m1)
                                new_msm = True
                            if torch.all(intersection == m2.bitmap):
                                to_remove[x][y].add(m1)
                                new_MSMs[x][y].add(m1 - m2)
                                new_msm = True
                        if first_iter: continue # first iteration last_MSM == curr_MSM
                        for m1, m2 in itertools.product(curr_MSMs[x][y], last_MSMs[x+dx][y+dy]):
                            if m1 == m2: continue
                            intersection = m1.bitmap * m2.bitmap
                            if torch.all(intersection == m1.bitmap):
                                to_remove[x+dx][y+dy].add(m2)
                                new_MSMs[x+dx][y+dy].add(m2 - m1)
                                new_msm = True
                            if torch.all(intersection == m2.bitmap):
                                to_remove[x][y].add(m1)
                                new_MSMs[x][y].add(m1 - m2)
                                new_msm = True
                        for m1, m2 in itertools.product(last_MSMs[x][y], last_MSMs[x+dx][y+dy]):
                            if m1 == m2: continue
                            intersection = m1.bitmap * m2.bitmap
                            if torch.all(intersection == m1.bitmap):
                                to_remove[x+dx][y+dy].add(m2)
                                new_MSMs[x+dx][y+dy].add(m2 - m1)
                                new_msm = True
                            if torch.all(intersection == m2.bitmap):
                                to_remove[x][y].add(m1)
                                new_MSMs[x][y].add(m1 - m2)
                                new_msm = True
                if not new_msm: continue
                __union2dset(MSMs, new_MSMs)
                first_iter = False
                __union2dset(last_MSMs, curr_MSMs)
                curr_MSMs = new_MSMs
                new_MSMs = [[set() for c in range(B.cols)] for r in range(B.rows)]
                
                for x in range(B.rows):
                    for y in range(B.cols):
                        for m in to_remove[x][y]:
                            last_MSMs[x][y].remove(m)
                            MSMs[x][y].remove(m)
        
            for x in range(B.rows):
                for y in range(B.cols):
                    to_remove = set()
                    for m in MSMs[x][y]:
                        if m.n == 0:
                            # If the number is 0, it can be set to don't care and it's neighbors can be cleared
                            to_clear = to_clear.union(__get_one_ind(m.bitmap))
                            if m.coord is not None:
                                x, y = m.coord
                                B.N[...,x,y] = -1
                            to_remove.add(m)
                            minecount = True
                        elif m.size == m.n:
                            # If the number is equal to the number of unknown cells, all the cells contain mines
                            to_flag = to_flag.union(__get_one_ind(m.bitmap))
                            to_remove.add(m)
                            minecount = True

    if B.n == torch.sum(b_msm):
        to_flag = to_clear.union(__get_one_ind(b_msm))
        return B, list(__flatten3d(MSMs)), None, to_clear, to_flag
    if B.n == 0:
        to_clear = to_clear.union(__get_one_ind(b_msm))
        return B, list(__flatten3d(MSMs)), None, to_clear, to_flag
    return B, list(__flatten3d(MSMs)), b_msm, to_clear, to_flag
    
def __flatten3d(l):
    return itertools.chain.from_iterable(itertools.chain.from_iterable(l))

def __union2dset(s1, s2):
    for x in range(len(s1)):
        for y in range(len(s1[x])):
            s1[x][y] = s1[x][y].union(s2[x][y])

def __reduce_board(B):
    for mine_coord in B.get_mines():
        B.subtract_mine(*mine_coord)

def __minecount_reduce(B, MSMs):
    to_clear = set()
    to_flag = set()
    for m in MSMs:
        if m.n == 0:
            # If the number is 0, it can be set to don't care and it's neighbors can be cleared
            to_clear = to_clear.union(__get_one_ind(m.bitmap))
            if m.coord is not None:
                x, y = m.coord
                B.N[...,x,y] = -1
        if m.size == m.n:
            # If the number is equal to the number of unknown cells, all the cells contain mines
            to_flag = to_flag.union(__get_one_ind(m.bitmap))

    # Make appropriate changes with known information
    for x,y in to_clear:
        B.set_clear(x, y, -1)
    for x,y in to_flag:
        B.set_mine(x, y)
        B.subtract_mine(x, y)

    if len(to_clear)==0 and len(to_flag)==0: 
        return None # return None to indicate no change
    return to_clear, to_flag

def __get_one_ind(F):
    return set(tuple(ind.numpy()) for ind in (F == 1).nonzero())