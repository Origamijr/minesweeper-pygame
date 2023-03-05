import torch

def get_one_ind(F: torch.Tensor):
    return list(tuple(ind.numpy()) for ind in (F == 1).nonzero())

def update_board(B, to_clear, to_flag):
    for coord in to_clear:
        B.set_clear(*coord)
    for coord in to_flag:
        B.set_mine(*coord)

def flatten_msm_dict(d):
    MSMs = []
    for coord in d:
        MSMs += d[coord]
    return MSMs