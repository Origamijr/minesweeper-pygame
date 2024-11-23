import torch
import torch.nn.functional as F
from core.bitmap import Bitmap
import itertools
from copy import deepcopy

    
class SolutionTableSparse:
    """
    TODO document
    """
    def __init__(self):
        self.table = torch.zeros(0, 0, dtype=torch.int8)
        self.coords = []
    
    def __len__(self):
        return self.table.shape[0]
    
    def __repr__(self):
        return ',\n'.join([str(torch.reshape(row, (self.rows, self.cols)).numpy()) for row in self.table])
    

    # ===== Methods to add and combine Solutions =====

    def add(self, solution:Bitmap):
        coords = solution.nonzero()
        self.table = F.pad(self.table, (0,len(coords-set(self.coords)),0,1))
        for coord in coords:
            if coord not in self.coords: self.coords.append(coord)
            self.table[-1,self.coords.index(coord)] = 1
    
    def add_mine_to_all_solutions(self, mine_coords):
        pass

    def concatenate(self, other):
        # Concatenates two tables into a new table
        pass
    
    def combine(self, other):
        # Combinatorically combine two tables into a new table. Assumes tables have nonoverlapping solutions
        pass
    
    # ===== Methods to remove mines or solutions

    def mask_coords(self, mask):
        pass
    

    # ===== Methods to count solutions =====

    def solution_intersection_mask(self, coords, val=1):
        # Return true/false size n bitmask indicating if solution as value val at coordinates given
        pass
    
    def get_solutions_with_coords(self, mine_coords=[], clear_coords=[]):
        # Returns table with summed rows fulfilling the given coordinate constraints
        pass
    
    def get_solutions_with_numbers(self, clear_coords, numbers):
        # Returns table with summed rows fulfilling the given coordinate constraints
        pass
    
    def get_num_solutions_with_mines(self):
        pass


class SolutionSet:
    """
    Wrapper around SolutionTable to provide a more understandable interface of operations on a set of solutions 
    """
    def __init__(self, bitmap:Bitmap=None):
        self.bitmap = bitmap # Indicates the area the set has solutions over
        self.solution_table = None if self.bitmap is None else SolutionTable(self.bitmap.shape)

    @staticmethod
    def powerset(bitmap:Bitmap):
        ss = SolutionSet()
        ss.bitmap = bitmap
        solution_table = SolutionTable(bitmap.shape)
        for soln in bitmap.powerset():
            solution_table.add(soln)
        ss.solution_table = solution_table
        return ss

    def __repr__(self) -> str:
        return 'Solutions over\n' + repr(self.bitmap) + '\n{\n'+repr(self.solution_table)+'\n}'
    
    def clone(self):
        return deepcopy(self)

    # ===== Methods to add and combine Solutions =====

    def add_solution(self, b:Bitmap):
        assert self.bitmap is not None and b * self.bitmap == b
        self.solution_table.add(b)
    
    def expand_solutions(self, new_bitmap:Bitmap, mine_coords):
        # Extend existing solutions with information of mines outside the set
        assert self.bitmap is None or (new_bitmap >= self.bitmap).all()
        if self.bitmap is None:
            # If no solutions for self, make new mines the only solution
            self.bitmap = new_bitmap
            new_mines = Bitmap.coords(self.bitmap.rows, self.bitmap.cols, mine_coords)
            self.solution_table = SolutionTable(self.bitmap.shape)
            self.add_solution(new_mines)
        else:
            # Otherwise append mines to existing solutions
            self.bitmap = new_bitmap
            self.solution_table.add_mine_to_all_solutions(mine_coords)
    
    @staticmethod
    def merge_solution_sets(ss1, ss2):
        # Merge two sets with the same bitmap
        if ss1.bitmap is None: return ss2
        if ss2.bitmap is None: return ss1
        assert ss1.bitmap == ss2.bitmap
        ss = SolutionSet(ss1.bitmap)
        ss.solution_table = ss1.solution_table.concatenate(ss2.solution_table)
        return ss
    
    @staticmethod
    def combine_solution_sets(ss1, ss2):
        # Combine two sets with disjoint bitmap, slow, but worth the time save
        if ss1.bitmap is None: return ss2
        if ss2.bitmap is None: return ss1
        assert not (ss1.bitmap * ss2.bitmap).any(), repr(ss1.bitmap) + '\n' + repr(ss2.bitmap)
        ss = SolutionSet(ss1.bitmap + ss2.bitmap)
        ss.solution_table = ss1.solution_table.combine(ss2.solution_table)
        return ss
    
    def add_region(self, bitmap:Bitmap):
        # Add all possible solutions involving a new unknown region
        # TODO consider adding a limit on number of mines if unknown region is large
        unknown_bitmap = bitmap - self.bitmap
        unknown_soln_table = SolutionTable(unknown_bitmap.shape)
        for soln in unknown_bitmap.powerset():
            unknown_soln_table.add(soln)
        if self.bitmap is None:
            self.bitmap = unknown_bitmap
            self.solution_table = unknown_soln_table
        else:
            self.bitmap += unknown_bitmap
            self.solution_table.combine(unknown_soln_table)


    # ===== Methods to reduce solutions =====

    def shrink_bitmap(self, bitmap:Bitmap):
        assert self.bitmap >= bitmap, 'Smaller solution area not a subset of original'
        self.bitmap = bitmap
        self.solution_table.mask_coords(bitmap)

    def remove_certainty(self):
        soln_count = self.solution_table.get_num_solutions_with_mines()
        clear_bitmap = Bitmap(*self.bitmap.shape, bitmap=(soln_count == 0)) * self.bitmap
        flag_bitmap = Bitmap(*self.bitmap.shape, bitmap=(soln_count == len(self.solution_table))) * self.bitmap
        self.solution_table.mask_coords((self.bitmap - clear_bitmap) - flag_bitmap)
        return clear_bitmap.nonzero(), flag_bitmap.nonzero()
    

    # ===== Methods to count solutions =====

    def get_solution_counts(self):
        # Wrapper for argumentless get_solution_counts_with_coords
        return self.get_solution_counts_with_coords()

    def get_solution_counts_with_coords(self, mine_coords=[], clear_coords=[]):
        # Return the number of solutions with coords that agree with the parameter, groupped by minecount
        solutions = self.solution_table.get_solutions_with_coords(mine_coords=mine_coords, clear_coords=clear_coords)
        mine_counts = torch.sum(solutions, dim=1, dtype=torch.int)
        if len(mine_counts) == 0: {0: 0}
        return {num_mines.item(): count.item() for num_mines, count in zip(*torch.unique(mine_counts, return_counts=True))}

    def get_solution_counts_with_numbers(self, clear_coords, numbers):
        # Return the number of solutions with numbers that agree with the parameter, groupped by minecount
        assert len(clear_coords) == len(numbers)
        solutions = self.solution_table.get_solutions_with_numbers(clear_coords, numbers)
        mine_counts = torch.sum(solutions, dim=1, dtype=torch.int)
        if len(mine_counts) == 0: {0: 0}
        return {num_mines.item(): count.item() for num_mines, count in zip(*torch.unique(mine_counts, return_counts=True))}

    def get_solution_counts_for_numbers(self, coord, mine_coords=[], clear_coords=[]):
        # Return the number of solutions with coords that agree with the parameter, groupped by number of mines around input coord
        solutions = self.solution_table.get_solutions_with_coords(mine_coords=mine_coords, clear_coords=clear_coords)
        if len(solutions) == 0: {0: 0}
        neighbor_mask = Bitmap.coords(self.bitmap.shape[0], self.bitmap.shape[1], [coord]).closure(count_center=False).flatten() != 0
        local_solutions = solutions[:,neighbor_mask]
        local_mine_counts = torch.sum(local_solutions, dim=1, dtype=torch.int)
        local_mine_counts_opts = torch.unique(local_mine_counts)
        mine_counts = torch.sum(solutions, dim=1, dtype=torch.int)
        counts = dict()
        for local_count in local_mine_counts_opts:
            local_count_mask = local_mine_counts == local_count
            mine_counts_filt = mine_counts[local_count_mask]
            counts[local_count.item()] = {num_mines.item(): count.item() for num_mines, count in zip(*torch.unique(mine_counts_filt, return_counts=True))}
        return counts
    
    def get_progress_with_coords(self, mine_coords=[], clear_coords=[]):
        solutions = self.solution_table.get_solutions_with_coords(mine_coords=mine_coords, clear_coords=clear_coords)
        if solutions.shape[0] == 0: return 0
        solution_counts = torch.sum(solutions, dim=0, dtype=torch.int)
        return torch.sum((solution_counts == 0) * self.bitmap) - len(clear_coords)
    
    def get_stats_with_coords(self, mine_coords=[], clear_coords=[]):
        solutions = self.solution_table.get_solutions_with_coords(mine_coords=mine_coords, clear_coords=clear_coords)
        mine_counts = torch.sum(solutions, dim=1, dtype=torch.int)
        if len(mine_counts) == 0: 
            counts = {0: 0}
        else:
            counts = {num_mines.item(): count.item() for num_mines, count in zip(*torch.unique(mine_counts, return_counts=True))}
        
        solution_counts = torch.sum(solutions, dim=0, dtype=torch.int)
        if solutions.shape[0] == 0: 
            progress = 0
        else:
            progress = torch.sum((solution_counts == 0) * self.bitmap) - len(clear_coords)
        return counts, progress