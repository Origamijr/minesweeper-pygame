import torch
import torch.nn.functional as F
from core.bitmap import Bitmap
import itertools
from copy import deepcopy

    
class SolutionTable:
    """
    ArrayList implementation of list of solutions with torch tensor as underlying representation
    Table is n x rc where each row is a solution and each column is a board coordinate
    """
    def __init__(self, board_shape, init_capacity=32):
        self.rows, self.cols = board_shape
        self.curr_element = -1
        self.table = torch.zeros(init_capacity, self.rows * self.cols, dtype=torch.int8)

    @staticmethod
    def from_table(board_shape, table):
        st = SolutionTable(board_shape)
        st.table = table
        st.curr_element = table.shape[0] - 1
    
    def __len__(self):
        return self.curr_element + 1
    
    def __repr__(self):
        return ',\n'.join([str(torch.reshape(row, (self.rows, self.cols)).numpy()) for row in self.table])
    
    def _coord2col(self, x, y):
        # convert (x,y) coordinate to column index
        return x * self.cols + y
    def _col2coord(self, col):
        return col // self.cols, col % self.cols
    def _expand(self, amount):
        # increase the capacity of the arraylist
        self.table = F.pad(self.table, (0,0,0,amount))
    def _trim_rows(self):
        # reduce capacity to the current size of the array
        self.table = self.table[:self.curr_element+1,:]
    def _capacity(self):
        return self.table.shape[0]

    def get_solutions(self):
        self._trim_rows()
        return self.table
    
    def get_counts(self):
        self._trim_rows()
        return {num_mines.item(): count.item() for num_mines, count in zip(*torch.unique(torch.sum(self.table, dim=1), return_counts=True))}

    def reduced_table(self, bitmap=None):
        # return a table with columns corresponding to an index list
        if bitmap is None:
            indices = list(torch.sum(self.table, dim=0).nonzero().flatten().numpy())
        else:
            indices = list(bitmap.flatten().nonzero().flatten().numpy())
        return self.table[:,indices], [self._col2coord(i) for i in indices]
    

    # ===== Methods to add and combine Solutions =====

    def add(self, solution:Bitmap):
        self.curr_element += 1
        if self.curr_element >= self._capacity(): self._expand(self._capacity())
        self.table[self.curr_element,:] = solution.flatten() # The new row is the solution flattend
    
    def add_mine_to_all_solutions(self, mine_coords):
        # Sets each column corresponding to each coordinate given to 1
        self.table[:,[self._coord2col(*coord) for coord in mine_coords]] = 1

    def concatenate(self, other):
        # Concatenates two tables into a new table
        self._trim_rows(), other._trim_rows()
        new_solution_bitmap = SolutionTable((self.rows, self.cols), init_capacity=0)
        new_solution_bitmap.table = torch.cat([self.table, other.table], dim=0)
        new_solution_bitmap.curr_element = new_solution_bitmap.table.shape[0] - 1
        return new_solution_bitmap
    
    def combine(self, other):
        # Combinatorically combine two tables into a new table. Assumes tables have nonoverlapping solutions
        self._trim_rows(), other._trim_rows()
        new_solution_bitmap = SolutionTable((self.rows, self.cols), init_capacity=0)
        new_solution_bitmap.table = self.table.repeat((len(other),1)) + other.table.repeat_interleave(len(self), dim=0)
        new_solution_bitmap.curr_element = new_solution_bitmap.table.shape[0] - 1
        return new_solution_bitmap
    
    # ===== Methods to remove mines or solutions

    def mask_coords(self, mask):
        self.table *= torch.tile(mask.flatten(), (self.table.shape[0],1))
        row_mask = torch.sum(self.table, dim=1).bool()
        self.table = torch.unique(self.table[row_mask,:], dim=0)
    

    # ===== Methods to get conditional solutions/counts =====

    def solution_intersection_mask(self, coords, val=1):
        # Return true/false size n bitmask indicating if solution as value val at coordinates given
        self._trim_rows()
        bitmap = self.table if val else 1-self.table
        return torch.prod(bitmap[:,[self._coord2col(*coord) for coord in coords]], dim=1, dtype=torch.int8)
    
    def get_solutions_with_coords(self, mine_coords=[], clear_coords=[]):
        # Returns table with summed rows fulfilling the given coordinate constraints
        mined_mask = self.solution_intersection_mask(mine_coords, val=1) > 0
        clear_mask = self.solution_intersection_mask(clear_coords, val=0) > 0
        return self.table[mined_mask * clear_mask,:]
    
    def get_solutions_with_numbers(self, clear_coords, numbers):
        # Returns table with summed rows fulfilling the given coordinate constraints
        clear_mask = self.solution_intersection_mask(clear_coords, val=0) > 0
        filtered_table = self.table[clear_mask,:]
        for coord, number in zip(clear_coords, numbers):
            x, y = coord
            neighbor_cols = []
            for dx, dy in itertools.product(range(-1,2), range(-1,2)):
                if 0 <= x+dx < self.rows and 0 <= y+dy < self.cols: neighbor_cols.append(self._coord2col((x+dx, y+dy)))
            filtered_table = filtered_table[torch.sum(filtered_table[:,neighbor_cols], dim=1, dtype=torch.int8) == number,:]
        return filtered_table
    
    def get_num_solutions_with_mines(self):
        # Returns 2D array with number of solutions with a mine at that location
        return torch.reshape(torch.sum(self.table, dim=0), (self.rows, self.cols))
    
    def get_solutions_with_num_mines(self, n):
        # Return rows of the table with n mines total
        return self.table[torch.sum(self.table, dim=1)==n,:]



class SolutionSet:
    """
    Wrapper around SolutionTable to provide a more understandable interface of operations on a set of solutions 
    """
    def __init__(self, bitmap:Bitmap=None):
        self.bitmap = bitmap # Indicates the area the set has solutions over
        self.solution_table = None if self.bitmap is None else SolutionTable(self.bitmap.shape)

    @staticmethod
    def powerset(bitmap:Bitmap):
        # Return a SolutionSet with a powerset number of mines
        ss = SolutionSet(bitmap=bitmap)
        for soln in bitmap.powerset():
            ss.solution_table.add(soln)
        return ss
    
    @staticmethod
    def combinations(bitmap:Bitmap, r):
        # Return a SolutionSet conatining all solutions in the bitmap with r mines
        ss = SolutionSet(bitmap=bitmap)
        for soln in bitmap.combinations(r):
            ss.solution_table.add(soln)
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
    
    def add_region(self, bitmap:Bitmap, n=None):
        # Add all possible solutions involving a new unknown region
        unknown_bitmap = bitmap - self.bitmap
        if self.bitmap is None:
            self.bitmap = unknown_bitmap
            if n is None:
                self.solution_table = SolutionSet.powerset(unknown_bitmap).solution_table
            else:
                self.solution_table = SolutionSet.combinations(unknown_bitmap, n).solution_table
            return
        
        if n is None:
            unknown_soln_table = SolutionSet.powerset(unknown_bitmap).solution_table
            self.bitmap += unknown_bitmap
            self.solution_table.combine(unknown_soln_table)
            return
        
        counts = self.solution_table.get_counts()
        table = SolutionTable(self.bitmap.shape)
        for num_mines in counts:
            r = n - num_mines
            if r < 0 or r > unknown_bitmap.sum(): continue
            subtable = SolutionTable.from_table(self.bitmap.shape, self.solution_table.get_solutions_with_num_mines(r))
            subtable.combine(SolutionSet.combinations(unknown_bitmap, r).solution_table)
            table.concatenate(subtable)
        self.solution_table = table






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
    
    def split_bitmap(self, bitmaps:list[Bitmap]):
        solutions = []
        for bitmap in bitmaps:
            solutions.append(self.clone())
            solutions[-1].shrink_bitmap(bitmap)
        return solutions
    

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
    
    def brute_force_win(self, verbose=0):
        # brute force win probability assuming all solutions are enumerated

        # get reduced table for simplicity
        table, coords = self.solution_table.reduced_table(self.bitmap)
        
        # create connectivity graph
        coord_graph = {c: [b for b in coords if -2<b[0]-c[0]<2 and -2<b[1]-c[1]<2 and b!=c] for c in coords}

        def _helper(table, coords, depth=0) -> dict[tuple,float]:
            w_prob = dict()
            # for each coordinate, calculate win probability recursively
            for i, coord in enumerate(coords):
                m_prob = torch.sum(table[:,i]) / table.shape[0]
                if m_prob == 1: continue # No need to evaluate further if coord is always mine
                if verbose >= 4: print(f'{"".join(["  "]*depth)}evaluating {coord}: {torch.sum(1-table[:,i])}/{table.shape[0]}')

                # Get the residual table from removing the current coordinate
                res_table = table[table[:,i]==0,:]
                res_table = torch.cat([res_table[:,:i],res_table[:,i+1:]], dim=1)
                res_coords = coords[:i]+coords[i+1:]

                # Get the number of neighbors the current coord has in the residual table
                neighbors = torch.zeros(res_table.shape[0])
                for ncoord in coord_graph[coord]:
                    if ncoord not in res_coords: continue
                    neighbors += res_table[:,res_coords.index(ncoord)]
                
                # Iterate over the possible number of neighor mines for the given coord
                w_prob[coord] = 0
                for n in set(neighbors.numpy()):
                    # Accumulate the safety value for each neighbor mine count
                    num_table = res_table[neighbors==n,:]
                    n_prob = num_table.shape[0] / res_table.shape[0]
                    if num_table.shape[0] == 1:
                        w_prob[coord] += n_prob # Base case when only one solution shows up after number chosen
                    else:
                        w_prob[coord] += n_prob * max(_helper(num_table, res_coords, depth+1).values())
                w_prob[coord] *= 1 - m_prob
            return w_prob
        
        w_prob_dict = _helper(table, coords)
        w_prob = torch.zeros(self.bitmap.shape)
        for coord, p in w_prob_dict.items():
            w_prob[coord] = p
        return w_prob

