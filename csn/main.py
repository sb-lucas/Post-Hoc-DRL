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

from tripletnet import CS_Tripletnet
from csn import ConditionalSimNet

from training_helpers.dataloader import TripletDataset
from training_helpers.arg_parser import parse_args
from training_helpers.checkpoint import save_checkpoint
from training_helpers.plot import plot_losses

def train(tnet, dataloader, optimizer, device, triplet_loss):
    tnet.train()
    epoch_loss = 0

    for batch_index, (anchor, negative, positive, condition) in enumerate(dataloader):
        print(f"Training - Batch {batch_index+1}/{len(dataloader)}", end='\r')

        #to device
        anchor, negative, positive, condition = anchor.to(device), negative.to(device), positive.to(device), condition.to(device)

        #forward pass
        dist_neg, dist_pos, mask_norm = tnet(anchor, negative, positive, condition)

        #calculate loss
        target = torch.ones_like(dist_neg)

        masks = F.relu(tnet.csn.masks.weight)
        M = F.normalize(masks, dim=1)
        I = torch.eye(M.shape[0], device=M.device)
        loss_ortho = ((M @ M.T - I)**2).sum()

        loss = triplet_loss(dist_neg, dist_pos, target) + (mask_norm * 1e-5) + (loss_ortho * 1e-3)

        #backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #accumulate loss
        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)

    return epoch_loss

@torch.no_grad()
def test(tnet, dataloader, device, triplet_loss):
    tnet.eval()
    epoch_loss = 0

    for batch_index, (anchor, negative, positive, condition) in enumerate(dataloader):
        print(f"Testing - Batch {batch_index+1}/{len(dataloader)}", end='\r')

        #to device
        anchor, negative, positive, condition = anchor.to(device), negative.to(device), positive.to(device), condition.to(device)

        #forward pass
        dist_neg, dist_pos, mask_norm = tnet(anchor, negative, positive, condition)

        #calculate loss
        target = torch.ones_like(dist_neg)
        loss = triplet_loss(dist_neg, dist_pos, target) + (mask_norm * 1e-5)

        #accumulate loss
        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)

    return epoch_loss

def main():
    #--------------------------------------------------Parsing and Logging--------------------------------------------------#
    args = parse_args()

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    experiment_name = f"{args.backbone}_{timestamp}"
    
    save_dir = os.path.join("runs", experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, "log.txt")
    settings_path = os.path.join(save_dir, "settings.txt")

    #--------------------------------------------------torch configs--------------------------------------------------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #-----------------------------------------------------Log Settings-------------------------------------------------#
    with open(settings_path, "w") as f:
        f.write("Experiment Settings:\n\n")
        f.write(f"Backbone Model: {args.backbone}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Epochs: {args.epochs}\n")

    #-------------------------------------------------Load Data----------------------------------------------#
    train_loader = DataLoader(TripletDataset(backbone=args.backbone, split='train'), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TripletDataset(backbone=args.backbone, split='val'), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TripletDataset(backbone=args.backbone, split='test'), batch_size=args.batch_size, shuffle=False)

    #-------------------------------------------------Load Model----------------------------------------------#
    if args.backbone == 'clap':
        dim_embed = 512
    elif args.backbone == 'musicnn':
        dim_embed = 200

    csn_model = ConditionalSimNet(n_conditions=3, embedding_size=dim_embed) 
    tnet = CS_Tripletnet(csn_model)

    csn_model = csn_model.to(device)
    tnet = tnet.to(device)

    #--------------------------------------------------Training Loop-----------------------------------------------#
    triplet_loss = nn.MarginRankingLoss(margin=0.2)
    optimizer = optim.Adam(tnet.parameters(), lr=args.lr)

    early_stop_patience = 5           
    early_stop_counter = 0 
    early_stop_threshold = 0.001

    train_losses = []
    val_losses = []
    best_loss = float('inf')

    with open(log_path, "w") as f:
        f.write("Starting Training\n")

    for epoch in range(1, args.epochs + 1):
        with open(log_path, "a") as f:
            f.write(f"\n-----------Epoch {epoch}-----------\n")
        
        # train for one epoch
        train_loss = train(tnet, train_loader, optimizer, device, triplet_loss)
        train_losses.append(train_loss)
        with open(log_path, "a") as f:
            f.write(f"\nTreino: {train_loss}\n")

        # evaluate on validation set
        val_loss = test(tnet, val_loader, device, triplet_loss)
        val_losses.append(val_loss)
        with open(log_path, "a") as f:
            f.write(f"\nValidação: {val_loss}\n")

        # remember best loss and save checkpoint
        is_best =  best_loss - val_loss > early_stop_threshold
        best_loss = min(val_loss, best_loss)
        state = { 
            'epoch': epoch, 
            'state_dict': tnet.state_dict(), 
            'best_loss': best_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
        save_checkpoint(state, is_best, save_dir)

        #---------- Early Stopping Logic ----------#
        if is_best:
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered after {early_stop_patience} epochs with no improvement.")
            break

    #-------------- Plotting Losses --------------#
    plot_losses(train_losses, val_losses, save_dir, label='Loss')

if __name__ == '__main__':
    main()    
