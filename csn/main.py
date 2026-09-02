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

from csn import ConditionalSimNet
from variations import OneLayer, MLP
from training_helpers.custom_collates import attention_train_collate_fn, attention_val_collate_fn, mean_train_collate_fn, mean_val_collate_fn
from training_helpers.semihard_loss import SemiHardTripletLoss
from training_helpers.dataloader import TripletDatasetTrain, TripletDatasetVal
from training_helpers.arg_parser import parse_args
from training_helpers.checkpoint import save_checkpoint
from training_helpers.plot import plot_losses
from training_helpers.emb_to_device import emb_to_device


def train(csn_model, dataloader, optimizer, device, semi_hard_loss_fn, l_mask=1e-5, l_ortho=1e-3, l_emb=1e-5):
    csn_model.train()
    epoch_loss = 0

    for batch_idx, (emb, i_vec, g_vec, m_vec, artist_ids) in enumerate(dataloader):
        #print(f"Training Online - Batch {batch_idx+1}/{len(dataloader)}", end="\r")

        emb = emb_to_device(emb, device)
        i_vec, g_vec, m_vec = (i_vec.to(device), g_vec.to(device), m_vec.to(device))
        artist_ids = artist_ids.to(device)
        B = emb[0].size(0)

        # 1. Condition Tensors: 0=genre, 1=instrument, 2=mood
        c_genre = torch.full((B,), 0, dtype=torch.long, device=device)
        c_inst = torch.full((B,), 1, dtype=torch.long, device=device)
        c_mood = torch.full((B,), 2, dtype=torch.long, device=device)

        # 2. Single forward pass per condition on the same unmasked embedding batch
        
        masked_emb_g, mask_g, emb_g = csn_model(emb, c_genre)
        masked_emb_i, mask_i, emb_i = csn_model(emb, c_inst)
        masked_emb_m, mask_m, emb_m = csn_model(emb, c_mood)

        # 3. Compute GPU semi-hard triplet loss per dimension
        triplet_loss_genre = semi_hard_loss_fn(masked_emb_g, g_vec, artist_ids)
        triplet_loss_inst = semi_hard_loss_fn(masked_emb_i, i_vec, artist_ids)
        triplet_loss_mood = semi_hard_loss_fn(masked_emb_m, m_vec, artist_ids)
        triplet_loss = triplet_loss_genre + triplet_loss_inst + triplet_loss_mood

        #average norms across the triplet
        mask_norm = (mask_g.norm(p=1, dim=1).mean() + mask_i.norm(p=1, dim=1).mean() + mask_m.norm(p=1, dim=1).mean())
        emb_norm = emb_g.norm(p=2, dim=1).mean() #since all embeddings are the same, we can just take one of them

        #orthogonality regularization
        masks = F.relu(csn_model.masks.weight)
        M = F.normalize(masks, dim=1)
        I = torch.eye(M.shape[0], device=M.device)
        loss_ortho = ((M @ M.T - I) ** 2).sum()

        # 5. Combined Loss
        total_loss = (triplet_loss + (mask_norm * l_mask) + (loss_ortho * l_ortho) + (emb_norm * l_emb))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
    

    return epoch_loss / len(dataloader)

@torch.no_grad()
def test(csn_model, dataloader, device, triplet_loss_fn, l_mask=1e-5, l_ortho=1e-3, l_emb=1e-5):
    csn_model.eval()
    epoch_loss = 0

    for batch_index, (anchor, negative, positive, condition) in enumerate(dataloader):
        #print(f"Testing - Batch {batch_index+1}/{len(dataloader)}", end='\r')

        #to device
        anchor = emb_to_device(anchor, device)
        negative = emb_to_device(negative, device)
        positive = emb_to_device(positive, device)
        condition = condition.to(device)

        #forward pass
        masked_emb_anchor, mask_anchor, emb_anchor  = csn_model(anchor, condition)
        masked_emb_negative, mask_negative, emb_negative = csn_model(negative, condition)
        masked_emb_positive, mask_positive, emb_positive = csn_model(positive, condition)

        #get distances
        dist_neg = F.pairwise_distance(masked_emb_anchor, masked_emb_negative, 2)
        dist_pos = F.pairwise_distance(masked_emb_anchor, masked_emb_positive, 2)
       
        #calculate loss
        target = torch.ones_like(dist_neg)
        triplet_loss = triplet_loss_fn(dist_neg, dist_pos, target) 
        epoch_loss += triplet_loss.item()

    epoch_loss /= len(dataloader)

    return epoch_loss

def main():
    load_dotenv()
    DATASETS_FOLDER = os.environ["DATASETS_FOLDER"]
    DATASETS_FEATURES_FOLDER = os.environ["DATASETS_FEATURES_FOLDER"]

    #--------------------------------------------------Parsing and Logging--------------------------------------------------#
    args = parse_args()
    n_runs = 1
    for run in range(n_runs):
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        learn_masks_str = "learned_mask" if args.learn_masks else "fixed_mask"
        #experiment_name = f"{args.backbone}_{args.pooling}_{args.variation}_{learn_masks_str}_run_{run}"
        experiment_name = f"{args.backbone}_{args.pooling}_{args.variation}_{learn_masks_str}_{timestamp}"
        
        save_dir = os.path.join("runs", experiment_name)
        
        print(f"\n\nStarting Experiment {experiment_name}/{n_runs}...\n")


        #if the directory exists, the experiment has already started
        if os.path.exists(save_dir):
            #if the final plot has been created, it has finished
            if any(f.endswith('.png') for f in os.listdir(save_dir)):
                print(f"Experiment {experiment_name} already completed. Skipping...")
                continue
            # Otherwise, the experiment is still running. then delete the directory and start over
            else:
                print(f"Experiment {experiment_name} already exists but not completed. Deleting and starting over...")
                import shutil
                shutil.rmtree(save_dir)

        #create/recreate the directory for the experiment and start/star over
        os.makedirs(save_dir, exist_ok=True)

        log_path = os.path.join(save_dir, "log.txt")
        settings_path = os.path.join(save_dir, "settings.txt")

        #--------------------------------------------------torch configs--------------------------------------------------#
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed = run
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        #-----------------------------------------------------Log Settings-------------------------------------------------#
        with open(settings_path, "w") as f:
            f.write("Experiment Settings:\n\n")
            f.write(f"Experiment Name: {experiment_name}\n")
            f.write(f"Run: {run+1}/{n_runs}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Learning Rate: {args.lr}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Backbone Model: {args.backbone}\n")
            f.write(f"Variation: {args.variation}\n")
            f.write(f"Pooling Method: {args.pooling}\n")
            f.write(f"Learn Masks: {args.learn_masks}\n")

        #-------------------------------------------------Load Data----------------------------------------------#
        if args.pooling == "attention":
            train_loader = DataLoader(TripletDatasetTrain(args.backbone, DATASETS_FOLDER, DATASETS_FEATURES_FOLDER, args.pooling, 'jamendo'), batch_size=args.batch_size, shuffle=True, collate_fn=attention_train_collate_fn)
            val_loader = DataLoader(TripletDatasetVal(args.backbone, DATASETS_FOLDER, DATASETS_FEATURES_FOLDER, args.pooling, 'jamendo', 'val'), batch_size=args.batch_size, shuffle=False, collate_fn=attention_val_collate_fn)
        elif args.pooling == "mean":
            train_loader = DataLoader(TripletDatasetTrain(args.backbone, DATASETS_FOLDER, DATASETS_FEATURES_FOLDER, args.pooling, 'jamendo'), batch_size=args.batch_size, shuffle=True, collate_fn=mean_train_collate_fn)
            val_loader = DataLoader(TripletDatasetVal(args.backbone, DATASETS_FOLDER, DATASETS_FEATURES_FOLDER, args.pooling, 'jamendo', 'val'), batch_size=args.batch_size, shuffle=False, collate_fn=mean_val_collate_fn)
        else:
            raise ValueError(f"Unsupported pooling method: {args.pooling}")

        #-------------------------------------------------Load Model----------------------------------------------#
        if args.backbone == 'clap':
            embedding_size = 768
        elif args.backbone == 'mert':
            embedding_size = 768
        elif args.backbone == 'muq':
            embedding_size = 1024
        else:
            raise ValueError(f"Unsupported backbone: {args.backbone}")

        if args.variation == "one_layer":
            embeddingnet = OneLayer(dim=embedding_size)
        elif args.variation == "mlp":
            embeddingnet = MLP(dim=embedding_size)
        elif args.variation == "none":
            embeddingnet = nn.Identity()  # Pass-through layer (0 parameters)
        else:
            raise ValueError(f"Unsupported variation: {args.variation}")

        csn_model = ConditionalSimNet(embeddingnet, args.n_conditions, embedding_size, args.pooling, args.learn_masks).to(device)

        #--------------------------------------------------Training Loop-----------------------------------------------#
        val_loss_fn = nn.MarginRankingLoss(margin=0.1)
        train_loss_fn = SemiHardTripletLoss(margin=0.1)
        optimizer = optim.Adam(csn_model.parameters(), lr=args.lr)

        # cosine decay from args.lr down to ~0 over the full training run
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        early_stop_patience = 20           
        early_stop_threshold = 0
        early_stop_counter = 0 

        train_losses = []
        val_losses = []
        best_loss = float('inf')

        with open(log_path, "w") as f:
            f.write("Starting Training\n")

        for epoch in range(1, args.epochs + 1):
            with open(log_path, "a") as f:
                f.write(f"\n-----------Epoch {epoch}-----------\n")
            
            # train for one epoch
            train_loss = train(csn_model, train_loader, optimizer, device, train_loss_fn)
            train_losses.append(train_loss)
            with open(log_path, "a") as f:
                f.write(f"\nTreino: {train_loss}\n")

            # evaluate on validation set
            val_loss = test(csn_model, val_loader, device, val_loss_fn)
            val_losses.append(val_loss)
            with open(log_path, "a") as f:
                f.write(f"\nValidação: {val_loss}\n")

            current_lr = optimizer.param_groups[0]['lr']
            with open(log_path, "a") as f:
                f.write(f"LR: {current_lr:.2e}\n")

            scheduler.step()

            # remember best loss and save checkpoint
            is_best =  best_loss - val_loss > early_stop_threshold
            best_loss = min(val_loss, best_loss)
            state = { 
                'epoch': epoch, 
                'state_dict': csn_model.state_dict(), 
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
                print(f"Experiment {experiment_name} Early stopping triggered after {early_stop_patience} epochs with no improvement.")
                break

        #-------------- Plotting Losses --------------#
        plot_losses(train_losses, val_losses, save_dir, label='Loss')

        # Free memory before next run
        del train_loader
        del val_loader
        del csn_model
        del optimizer
        del scheduler
        del train_loss_fn
        del val_loss_fn

        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()    
