import os
from datetime import datetime
import random
import sys
import pandas as pd
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

from triplet_score import triplet_score
from dim_sim_score import evaluate_dimsim
from mask_evaluation import mask_quality
from arg_parser import parse_args

sys.path.append("/home/ldsbrandao/exp/cerri/csn")
from training_helpers.dataloader import TripletDataset
from tripletnet import CS_Tripletnet
from csn import ConditionalSimNet

def main():
    args = parse_args()

    #device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load data
    test_loader = DataLoader(TripletDataset(backbone=args.backbone, split='test'), batch_size=args.batch_size, shuffle=False)

    #choose the embedding dimension based on the backbone
    if args.backbone == 'clap':
        dim_embed = 512
    elif args.backbone == 'musicnn':
        dim_embed = 200
    
    #initialize the CSN model and the triplet network
    csn_model = ConditionalSimNet(n_conditions=3, embedding_size=dim_embed) 
    tnet = CS_Tripletnet(csn_model)

    #load checkpoint
    checkpoint_folder = f"/home/ldsbrandao/exp/cerri/csn/runs/{args.path}"
    model_weights_complete_path = os.path.join(checkpoint_folder, "model_best.pth.tar")
    checkpoint = torch.load(model_weights_complete_path, map_location=device)
    tnet.load_state_dict(checkpoint['state_dict'])

    #to device
    csn_model = csn_model.to(device)
    tnet = tnet.to(device)

    #get the learned masks from the CSN model
    masks = F.relu(tnet.csn.masks.weight).to(device) #shape (3, embedding_dim)

    #Calculate triplet score of each mask and the backbone over test triplets (separeted per triplet type)
    triplet_acc = triplet_score(test_loader, device, masks)
    print(f"Triplet Accuracy:")
    df = pd.DataFrame(triplet_acc).T
    df.to_csv(os.path.join(checkpoint_folder, "triplet_scores.csv"), index=True)

    #calculate sparsity and overlap of the learned masks
    sparsity_dict, overlap_dict = mask_quality(masks)
    sparsity_df = pd.DataFrame.from_dict(sparsity_dict, orient='index', columns=['sparsity'])
    overlap_df = pd.DataFrame.from_dict(overlap_dict, orient='index', columns=['overlap'])
    sparsity_df.to_csv(os.path.join(checkpoint_folder, "mask_sparsity.csv"))
    overlap_df.to_csv(os.path.join(checkpoint_folder, "mask_overlap.csv"))

    #calculate the dim sim score of each mask and the backbone over the dim sim dataset
    dimsim_acc = evaluate_dimsim(args.backbone, masks)
    dimsim_df = pd.DataFrame.from_dict(dimsim_acc, orient='index', columns=['dimsim_accuracy'])
    dimsim_df.to_csv(os.path.join(checkpoint_folder, "dimsim_scores.csv"))



if __name__ == "__main__":
    main()