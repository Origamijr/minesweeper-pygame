import torch

def get_one_ind(F: torch.Tensor):
    return set(tuple(ind.numpy()) for ind in (F == 1).nonzero())