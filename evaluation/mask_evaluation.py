import os
from datetime import datetime
import random
import sys
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import argparse
import librosa

import torch
import torch.nn.functional as F

condition_mapping = {
    0: "instrument",
    1: "genre",
    2: "theme"
}

#Function to calculate the sparsity (percentage of active dimensions (>0)) per mask
#and the overlap between masks (pairwise cosine similarity)
@torch.no_grad()
def mask_quality(masks):

    sparsity_dict = {}

    # Sparsity
    for i, mask in enumerate(masks):
        cond_name = condition_mapping[i]

        # fraction of dimensions that are active
        sparsity_dict[cond_name] = (mask > 0).float().mean().item()

    # Pairwise cosine similarities
    overlap_dict = {}

    n_masks = masks.shape[0]

    for i in range(n_masks):
        for j in range(i + 1, n_masks):

            cond_i = condition_mapping[i]
            cond_j = condition_mapping[j]

            similarity = F.cosine_similarity(
                masks[i].unsqueeze(0),
                masks[j].unsqueeze(0),
                dim=1
            ).item()

            overlap_dict[f"{cond_i}-{cond_j}"] = similarity

    return sparsity_dict, overlap_dict