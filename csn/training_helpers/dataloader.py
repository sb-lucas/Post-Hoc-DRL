import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
import pandas as pd
import numpy as np
import torch.nn.functional as F
import librosa
import sys, os
import random
import math
import pickle

class TripletDataset(Dataset):
    def __init__(self, backbone, split='train'):
        
        self.condition_mapping = {
            'instrument':0,
            'genre':1,
            'theme':2
        }

        if backbone == 'musicnn':
            self.embeddings_folder = "/home/ldsbrandao/exp/diego/diego/final/data/extracted_features/musicnn/original"
        elif backbone == 'clap':
            self.embeddings_folder = "/home/ldsbrandao/exp/diego/diego/final/data/extracted_features/clap_global/original"

        triplets_path = f'/home/ldsbrandao/exp/cerri/dataset/{split}_triplets.csv'
        self.triplets_df = pd.read_csv(triplets_path)
        
        self.all_songs = pd.concat([self.triplets_df["anchor"], self.triplets_df["positive"], self.triplets_df["negative"]]).unique()

        #pre loading all audio embeddings
        os.makedirs("cache", exist_ok=True)

        self.audio_cache = {}
        cache_file_path = f"cache/cache_{backbone}_{split}.pt"
        if os.path.exists(cache_file_path):
            print("Loading pre-computed audio cache...")
            self.audio_cache = torch.load(cache_file_path, map_location="cpu")
        else:
            for i, song in enumerate(self.all_songs):
                print(f"Pre-loading audio embeddings: {i+1}/{len(self.all_songs)}. song = {song}: ", end='\r')
                song_id = song.split('.')[0]
                embedding_path = os.path.join(self.embeddings_folder, f"{song_id}.npy")
                embedding = torch.from_numpy(np.load(embedding_path)).float().squeeze()
                embedding = F.normalize(embedding, dim=-1)
                self.audio_cache[song_id] = (embedding)

            #save cache
            torch.save(self.audio_cache, cache_file_path)

    def get_num_songs(self):
        return len(self.all_songs)

    def __len__(self):
        return len(self.triplets_df)

    def __getitem__(self, idx):
        row = self.triplets_df.iloc[idx]

        anchor = row['anchor'].split('.')[0]
        positive = row['positive'].split('.')[0]
        negative = row['negative'].split('.')[0]
        
        emb_anchor = self.audio_cache[anchor]
        emb_positive = self.audio_cache[positive]
        emb_negative = self.audio_cache[negative]

        return emb_anchor, emb_negative, emb_positive, self.condition_mapping[row['condition']]