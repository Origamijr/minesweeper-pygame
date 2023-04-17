import torch
import itertools

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

def merge_disjoint_counts(counts1, counts2):
    # Merge counts of dijoint sets
    if len(counts1) == 0: return counts2
    if len(counts2) == 0: return counts1
    counts = dict()
    for (n1, count1), (n2, count2) in itertools.product(counts1.items(), counts2.items()):
        if not n1+n2 in counts: counts[n1+n2] = 0
        counts[n1+n2] += count1*count2
    return counts