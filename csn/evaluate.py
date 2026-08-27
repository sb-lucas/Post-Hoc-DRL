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
from dotenv import load_dotenv
import pandas as pd

from csn import ConditionalSimNet
from variations import OneLayer, MLP
from training_helpers.custom_collates import attention_train_collate_fn, attention_val_collate_fn, mean_train_collate_fn, mean_val_collate_fn
from training_helpers.semihard_loss import SemiHardTripletLoss
from training_helpers.dataloader import TripletDatasetTrain, TripletDatasetVal
from training_helpers.arg_parser import parse_args
from training_helpers.checkpoint import save_checkpoint
from training_helpers.plot import plot_losses
from training_helpers.emb_to_device import emb_to_device

'''
get the triplet score of a model on the triplet dataset.
For a given mask (trained to select instrument, or genre, or theme),
we will use it to calculate the triplet score of all triplets.
We also use mask=1*emb_size, to represent the backbone model.
We use a random mask too, randomly initizalied per run.

This way, we create a 5x4 matrix, where the columns represent the condition of the triplet (instrument, genre, theme, overall), 
and the rows represent the mask used (instrument, genre, theme, or none). This way, the diagonal
should dominate, as the best mask for a given condition should be the one trained for that condition.

The overall condition means that we do not consider the condition, 
we just calculate the triplet score the same way for every triplet. 
This is equivalent to micro average.

For the cases when mask is trained, we also calculate the sparsity of the mask 
(percentage of active dimensions (>0)) and the overlap between masks (pairwise cosine similarity).
'''

condition_mapping = {
    0: "genre",
    1: "instrument",
    2: "mood"
}
reverse_condition_mapping = {
    'genre':0,
    'instrument':1,
    'mood':2
}

def triplet_score(csn_model, dataloader, device, masks, run, n_conditions, embedding_size):
    acc = {"backbone": None, "random": None, "instrument": None, "genre": None, "mood": None}

    for condition in acc.keys():
        if condition == 'random':
            #for random, we generate a random mask with the same shape as the learned masks, using the run as seed number
            torch.manual_seed(run)
            mask = torch.rand(embedding_size, device=device)
        elif condition == 'backbone':
            # No masking for backbone, use a mask of ones
            mask = torch.ones_like(masks[0]).to(device)  
        else:
            # Use the learned mask for the specific condition
            mask = masks[reverse_condition_mapping[condition]]
            
        acc[condition] = __triplet_score__(csn_model, dataloader, device, mask)
    
    return acc

@torch.no_grad()    
def __triplet_score__(csn_model, dataloader, device, mask):
    csn_model.eval()

    correct = {
        "overall": 0,
        "instrument": 0,
        "genre": 0,
        "mood": 0
    }

    for batch_index, (anchor, negative, positive, condition) in enumerate(dataloader):
        print(f"Testing - Batch {batch_index+1}/{len(dataloader)}", end='\r')

        #to device
        anchor = emb_to_device(anchor, device)
        negative = emb_to_device(negative, device)
        positive = emb_to_device(positive, device)
        condition = condition.to(device)

        #forward pass through the model to get embeddings
        _, _, emb_anchor  = csn_model(anchor, condition)
        _, _, emb_negative = csn_model(negative, condition)
        _, _, emb_positive = csn_model(positive, condition)

        #mask according to a fixed mask, not necessarily the adequated to the condition of the triplet
        masked_emb_anchor = emb_anchor * mask
        masked_emb_negative = emb_negative * mask
        masked_emb_positive = emb_positive * mask

        #forward pass
        dist_neg = F.pairwise_distance(masked_emb_anchor, masked_emb_negative, 2)
        dist_pos = F.pairwise_distance(masked_emb_anchor, masked_emb_positive, 2)
        
        # Boolean tensor indicating whether each triplet is correctly ordered
        pred = (dist_neg > dist_pos)

        # overall
        correct["overall"] += pred.sum().item()

        # per condition
        for i, cond in enumerate(condition):
            cond_name = condition_mapping[cond.item()]
            correct[cond_name] += pred[i].item()
    
    acc = {k: correct[k] / dataloader.dataset.get_len(k) for k in correct}

    return acc  

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

def main():
    load_dotenv()
    DATASETS_FEATURES_FOLDER = os.environ["DATASETS_FEATURES_FOLDER"]
    DATASETS_FOLDER = os.environ["DATASETS_FOLDER"]           
    N_CONDITIONS = 3

    #--------------------------------------------------Parsing and Logging--------------------------------------------------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for pooling in ["attention", "mean"]:
        for backbone in ["muq", "clap", "mert"]:
            if backbone == 'clap':
                embedding_size = 768
            elif backbone == 'mert':
                embedding_size = 768
            elif backbone == 'muq':
                embedding_size = 1024

            print(f"Loading DataLoaders for backbone={backbone} and pooling={pooling}...")

            if pooling == "attention":
                jamendo_loader = DataLoader(TripletDatasetVal(backbone, DATASETS_FOLDER, DATASETS_FEATURES_FOLDER, pooling, 'jamendo', 'test'), batch_size=256, shuffle=False, collate_fn=attention_val_collate_fn)
                mtt_loader = DataLoader(TripletDatasetVal(backbone, DATASETS_FOLDER, DATASETS_FEATURES_FOLDER, pooling, 'magnatagatune', 'test'), batch_size=256, shuffle=False, collate_fn=attention_val_collate_fn)
            else:
                jamendo_loader = DataLoader(TripletDatasetVal(backbone, DATASETS_FOLDER, DATASETS_FEATURES_FOLDER, pooling, 'jamendo', 'test'), batch_size=256, shuffle=False, collate_fn=mean_val_collate_fn)
                mtt_loader = DataLoader(TripletDatasetVal(backbone, DATASETS_FOLDER, DATASETS_FEATURES_FOLDER, pooling, 'magnatagatune', 'test'), batch_size=256, shuffle=False, collate_fn=mean_val_collate_fn)
            
            for variation in ["none", "one_layer", "mlp"]:
                if variation == "one_layer":
                    embeddingnet = OneLayer(dim=embedding_size)
                elif variation == "mlp":
                    embeddingnet = MLP(dim=embedding_size)
                elif variation == "none":
                    embeddingnet = nn.Identity()  # Pass-through layer (0 parameters)

                if variation == "none":
                    learn_masks_options = [True]
                else:
                    learn_masks_options = [True, False] 

                for learn_masks in learn_masks_options:
                    n_runs = 5
                    for run in range(n_runs):
                        learn_masks_str = "learned_mask" if learn_masks else "fixed_mask"
                        experiment_name = f"{backbone}_{pooling}_{variation}_{learn_masks_str}_run_{run}"
                        checkpoint_folder = os.path.join("runs", experiment_name)

                        print(f"\n\nEvaluating: {experiment_name} on Jamendo and MTT datasets")

                        if not os.path.exists(checkpoint_folder):
                            print(f"Skipping missing checkpoint directory: {checkpoint_folder}")
                            continue

                        if not any(f.endswith('.png') for f in os.listdir(checkpoint_folder)):
                            print(f"Experiment not completed: {checkpoint_folder}")
                            continue

                        save_jamendo = os.path.join(checkpoint_folder, "triplet_scores_jamendo.csv")
                        save_mtt = os.path.join(checkpoint_folder, "triplet_scores_mtt.csv")

                        save_sparsity = os.path.join(checkpoint_folder, "mask_sparsity.csv")
                        save_overlap = os.path.join(checkpoint_folder, "mask_overlap.csv")

                        #if already evaluated, skip 
                        if os.path.exists(save_jamendo) and os.path.exists(save_mtt):
                            if learn_masks and (os.path.exists(save_sparsity) and os.path.exists(save_overlap)):
                                print(f"Already evaluated mask quality: {experiment_name}")
                                continue
                            elif not learn_masks:
                                print(f"Already evaluated triplet scores: {experiment_name}")
                                continue

                        csn_model = ConditionalSimNet(embeddingnet, N_CONDITIONS, embedding_size, pooling, learn_masks).to(device)
                        checkpoint_path = os.path.join(checkpoint_folder, "model_best.pth.tar")
                        checkpoint = torch.load(checkpoint_path, map_location=device)
                        csn_model.load_state_dict(checkpoint['state_dict'])
                        masks = F.relu(csn_model.masks.weight).to(device)  #get the learned masks from the CSN model

                        #Calculate triplet score of each mask and the backbone over test triplets (separeted per triplet type)
                        triplet_acc_jamendo = triplet_score(csn_model, jamendo_loader, device, masks, run, N_CONDITIONS, embedding_size)
                        triplet_acc_mtt = triplet_score(csn_model, mtt_loader, device, masks, run, N_CONDITIONS, embedding_size)

                        print(f"Triplet Accuracy:")
                        df_triplet_acc_jamendo = pd.DataFrame(triplet_acc_jamendo).T
                        df_triplet_acc_mtt = pd.DataFrame(triplet_acc_mtt).T
                        
                        df_triplet_acc_jamendo.to_csv(save_jamendo, index=True)
                        df_triplet_acc_mtt.to_csv(save_mtt, index=True)

                        if learn_masks:
                            #calculate sparsity and overlap of the learned masks
                            sparsity_dict, overlap_dict = mask_quality(masks)
                            sparsity_df = pd.DataFrame.from_dict(sparsity_dict, orient='index', columns=['sparsity'])
                            overlap_df = pd.DataFrame.from_dict(overlap_dict, orient='index', columns=['overlap'])
                            sparsity_df.to_csv(save_sparsity)
                            overlap_df.to_csv(save_overlap)

                        #free model of each run to save memory, as we are running many runs in a loop
                        del csn_model
                        del masks

            #each cache depends only on backbone and pooling, so we can free memory after each backbone/pooling combination
            del jamendo_loader
            del mtt_loader
    
            import gc
            gc.collect()
    
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()