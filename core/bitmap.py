import torch
import torch.nn.functional as F
import functools

NEIGHBOR_KERNEL = torch.tensor([[[[1.,1.,1.],
                                  [1.,0.,1.],
                                  [1.,1.,1.]]]], dtype=torch.int8)

NEIGHBOR_PLUS_KERNEL = torch.tensor([[[[1.,1.,1.],
                                       [1.,1.,1.],
                                       [1.,1.,1.]]]], dtype=torch.int8)

class Bitmap:
    # Just a wrapper class for an array of 1's and 0's in case a more efficient implementation is required in the future
    # Name is a misnomer atm since it can contain values -128 to 127. May eventually create true bitmap class
    def __init__(self, rows, cols, bitmap:torch.Tensor=None):
        self.rows, self.cols = rows, cols
        self.bitmap = bitmap.clone().to(torch.int8) if bitmap is not None else torch.zeros((1,1,rows,cols), dtype=torch.int8)

    @staticmethod
    def ones(rows, cols):
        return Bitmap(rows, cols, bitmap=torch.ones((1,1,rows,cols), dtype=torch.int8))

    @staticmethod
    def neg_ones(rows, cols):
        return Bitmap(rows, cols, bitmap=-torch.ones((1,1,rows,cols), dtype=torch.int8))

    def clone(self):
        return Bitmap(self.rows, self.cols, bitmap=self.bitmap)

    def __getitem__(self, key):
        if not isinstance(key, tuple): key = (key,)
        key = (slice(None),slice(None)) + (key)
        return self.bitmap[key]

    def __setitem__(self, key, value):
        if not isinstance(key, tuple): key = (key,)
        key = (slice(None),slice(None)) + (key)
        self.bitmap[key] = value
    
    def __eq__(self, other):
        return torch.all(self.bitmap == other.bitmap)
    
    def __gt__(self, other):
        return Bitmap(self.rows, self.cols, bitmap=(self.bitmap > other.bitmap))
    
    def __repr__(self):
        return str(self.bitmap[0,0,...].numpy())
    
    def __hash__(self):
        return hash(repr(self))
    
    def __add__(self, other): # union
        return Bitmap(self.rows, self.cols, bitmap=torch.bitwise_or(self.bitmap, other.bitmap))
    
    def __neg__(self): # complement
        return Bitmap(self.rows, self.cols, bitmap=(1-self.bitmap))

    def __sub__(self, other): # intersection with complement
        return self * (-other)
    
    def __mul__(self, other): # intersection
        return Bitmap(self.rows, self.cols, bitmap=torch.bitwise_and(self.bitmap, other.bitmap))

    def add(self, other):
        return Bitmap(self.rows, self.cols, bitmap=(self.bitmap + other.bitmap))
    
    def sub(self, other):
        return Bitmap(self.rows, self.cols, bitmap=(self.bitmap - other.bitmap))
    
    def mul(self, other):
        return Bitmap(self.rows, self.cols, bitmap=(self.bitmap * other.bitmap))

    def neighbor_count_map(self, count_center=False):
        # Returns a bitmap containing the sum o fthe neighborhood of each cell
        return Bitmap(self.rows, self.cols, bitmap=F.conv2d(F.pad(self.bitmap, (1,1,1,1), value=0), NEIGHBOR_PLUS_KERNEL if count_center else NEIGHBOR_KERNEL))
    
    def closure(self, count_center=True):
        # The bitmap containing the union of the neighborhoods
        return Bitmap(self.rows, self.cols, bitmap=self.neighbor_count_map(count_center=count_center).bitmap > 0)
    
    def all(self):
        return torch.all(self.bitmap)
    
    def any(self):
        return torch.any(self.bitmap)

    def nonzero(self):
        coords = list(tuple(ind.numpy()) for ind in self.bitmap[0,0,...].nonzero())
        coords.sort()
        return coords

    def sum(self):
        return torch.sum(self.bitmap).item()
    
    def get_mask(self, value:int):
        return Bitmap(self.rows, self.cols, bitmap=(self.bitmap == value))