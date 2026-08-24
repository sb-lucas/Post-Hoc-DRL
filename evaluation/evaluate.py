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
        embedding_size = 768
    elif args.backbone == 'mert':
        embedding_size = 768
    elif args.backbone == 'muq':
        embedding_size = 1024
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")

#     state = { 
#     'epoch': epoch, 
#     'state_dict': csn_model.state_dict(), 
#     'best_loss': best_loss,
#     'train_losses': train_losses,
#     'val_losses': val_losses,
#     'variation': args.variation,        # add this
#     'embedding_size': embedding_size,   # add this
#     'pooling': args.pooling,            # add this
#     'n_conditions': args.n_conditions,  # add this
#     'learn_masks': args.learn_masks,    # add this
# }

# checkpoint = torch.load(checkpoint_path, map_location=device)

# if checkpoint['variation'] == "one_layer":
#     embeddingnet = OneLayer(dim=checkpoint['embedding_size'])
# elif checkpoint['variation'] == "mlp":
#     embeddingnet = MLP(dim=checkpoint['embedding_size'])
# elif checkpoint['variation'] == "none":
#     embeddingnet = nn.Identity()

# csn_model = ConditionalSimNet(embeddingnet, checkpoint['n_conditions'], checkpoint['embedding_size'], checkpoint['pooling'], checkpoint['learn_masks']).to(device)
# csn_model.load_state_dict(checkpoint['state_dict'])

    
    #initialize the CSN model and the triplet network
    csn_model = ConditionalSimNet(embeddingnet, args.n_conditions, embedding_size, args.pooling, args.learn_masks).to(device)

    #load checkpoint
    checkpoint_folder = os.path.join(RUNS_PATH, args.path)
    model_weights_complete_path = os.path.join(checkpoint_folder, "model_best.pth.tar")
    checkpoint = torch.load(model_weights_complete_path, map_location=device)
    csn_model.load_state_dict(checkpoint['state_dict'])

    #to device
    csn_model = csn_model.to(device)

    #get the learned masks from the CSN model
    masks = F.relu(csn.masks.weight).to(device) #shape (3, embedding_dim)

    #Calculate triplet score of each mask and the backbone over test triplets (separeted per triplet type)
    triplet_acc = triplet_score(test_loader, device, masks)
    print(f"Triplet Accuracy:")
    df = pd.DataFrame(triplet_acc).T
    df.to_csv(os.path.join(checkpoint_folder, "triplet_scores.csv"), index=True)

    if args.learn_mask:
        #calculate sparsity and overlap of the learned masks
        sparsity_dict, overlap_dict = mask_quality(masks)
        sparsity_df = pd.DataFrame.from_dict(sparsity_dict, orient='index', columns=['sparsity'])
        overlap_df = pd.DataFrame.from_dict(overlap_dict, orient='index', columns=['overlap'])
        sparsity_df.to_csv(os.path.join(checkpoint_folder, "mask_sparsity.csv"))
        overlap_df.to_csv(os.path.join(checkpoint_folder, "mask_overlap.csv"))

if __name__ == "__main__":
    main()