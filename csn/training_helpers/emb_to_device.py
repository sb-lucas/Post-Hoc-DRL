import torch

def emb_to_device(emb_pair, device):
    x, mask = emb_pair
    x = x.to(device)
    if mask is not None:
        mask = mask.to(device)
    return x, mask