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


'''
get the triplet score of a model on the triplet dataset.
For a given mask (trained to select instrument, or genre, or theme),
we will use it to calculate the triplet score of all triplets.
We also use mask=None, to represent the backbone model.

This way, we create a 4x4 matrix, where the rows represent the condition of the triplet (instrument, genre, theme, overall), 
and the columns represent the mask used (instrument, genre, theme, or none). This way, the diagonal
should dominate, as the best mask for a given condition should be the one trained for that condition.

The overall condition means that we do not consider the condition, 
we just calculate the triplet score the same way for every triplet.
'''

condition_mapping = {
    0: "instrument",
    1: "genre",
    2: "theme"
}

reverse_condition_mapping = {
    "instrument": 0,
    "genre": 1,
    "theme": 2
}

def triplet_score(dataloader, device, masks):
    acc = {"backbone": 0, "instrument": 0, "genre": 0, "theme": 0}

    for condition in acc.keys():
        #mask the audio according to the condition, if not backbone
        if condition != 'backbone':
            mask = masks[reverse_condition_mapping[condition]]
        else:
            mask = None

        acc[condition] = __triplet_score__(dataloader, device, mask)
    
    return acc

@torch.no_grad()    
def __triplet_score__(dataloader, device, mask):
    correct = {
        "overall": 0,
        "instrument": 0,
        "genre": 0,
        "theme": 0
    }

    total = {
        "overall": 0,
        "instrument": 0,
        "genre": 0,
        "theme": 0
    }

    for batch_index, (anchor, negative, positive, condition) in enumerate(dataloader):
        print(f"Testing - Batch {batch_index+1}/{len(dataloader)}", end='\r')

        anchor = anchor.to(device)
        negative = negative.to(device)
        positive = positive.to(device)
        condition = condition.to(device)

        if mask is not None:
            anchor = anchor * mask
            negative = negative * mask
            positive = positive * mask

        #forward pass
        dist_neg = F.pairwise_distance(anchor, negative, 2)
        dist_pos = F.pairwise_distance(anchor, positive, 2)
        
        # Boolean tensor indicating whether each triplet is correctly ordered
        pred = (dist_neg > dist_pos)

        # overall
        correct["overall"] += pred.sum().item()
        total["overall"] += len(pred)

        # per condition
        for i, cond in enumerate(condition):
            cond_name = condition_mapping[cond.item()]

            correct[cond_name] += pred[i].item()
            total[cond_name] += 1

    acc = {
        k: correct[k] / total[k] if total[k] > 0 else 0
        for k in correct
    }

    return acc
