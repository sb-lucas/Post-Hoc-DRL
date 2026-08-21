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
import torch
import torch.nn as nn
import ast

class TripletDataset(Dataset):
    def __init__(self, backbone, DATASETS_FOLDER, all_songs, split, pooling):
        if backbone == "mert":
            self.embeddings_folder = f"{DATASETS_FOLDER}/jamendo/downsampled_mert_features"
        elif backbone == "clap":
            self.embeddings_folder = f"{DATASETS_FOLDER}/jamendo/downsampled_clap_features"
        elif backbone == "muq":
            self.embeddings_folder = f"{DATASETS_FOLDER}/jamendo/downsampled_muq_features"
        
        self.condition_mapping = {
            'genre':0,
            'instrument':1,
            'mood':2
        }

        #pre loading all audio embeddings
        os.makedirs("cache", exist_ok=True)

        self.audio_cache = {}
        cache_file_path = f"cache/cache_{backbone}_{pooling}_{split}.pt"
        if os.path.exists(cache_file_path):
            print("Loading pre-computed audio cache...")
            print("cache_file_path = ", cache_file_path)
            self.audio_cache = torch.load(cache_file_path, map_location="cpu", weights_only=True)
        else:
            if pooling == "attention":
                #load the way it is; collate_fn handles padding and CSN does pooling
                for i, song in enumerate(all_songs):
                    print(f"song = {song}: ", end='\r')
                    print(f"Pre-loading audio embeddings: {i+1}/{len(all_songs)}. song = {song}: ", end='\r')
                    song_id = os.path.splitext(song)[0]  # Remove the file extension to get the song ID
                    embedding_path = os.path.join(self.embeddings_folder, f"{song_id}.npy")
                    embedding = torch.from_numpy(np.load(embedding_path)).float().squeeze()
                    self.audio_cache[song_id] = (embedding)
            elif pooling == "mean":
                #load, but pool
                for i, song in enumerate(all_songs):
                    print(f"song = {song}: ", end='\r')
                    print(f"Pre-loading audio embeddings: {i+1}/{len(all_songs)}. song = {song}: ", end='\r')
                    song_id = os.path.splitext(song)[0]  # Remove the file extension to get the song ID
                    embedding_path = os.path.join(self.embeddings_folder, f"{song_id}.npy")
                    embedding = torch.from_numpy(np.load(embedding_path)).float().squeeze()
                    embedding = embedding.mean(dim=0) #mean pooling
                    self.audio_cache[song_id] = (embedding)

            #save cache
            print("\n\nsaving audio cache to disk...")
            torch.save(self.audio_cache, cache_file_path)


class TripletDatasetTrain(TripletDataset):
    def __init__(self, backbone, DATASETS_FOLDER, pooling):
        self.pooling = pooling

        self.df = pd.read_csv(f"{DATASETS_FOLDER}/jamendo/jamendo_train_split.csv")
        #get a debug sample of 100
        self.df = self.df.sample(n=100, random_state=42).reset_index(drop=True)


        self.artist_to_idx = {artist_id: idx for idx, artist_id in enumerate(self.df['ARTIST_ID'].unique())}

        all_songs = self.df["PATH"].tolist()


        super().__init__(backbone, DATASETS_FOLDER, all_songs, "train", pooling)
        
        # Unique tag vocabularies per category
        self.genre_vocab = sorted(list(set([t for tags in self.df["GENRE_TAGS"] for t in ast.literal_eval(tags)])))
        self.inst_vocab = sorted(list(set([t for tags in self.df["INSTRUMENT_TAGS"] for t in ast.literal_eval(tags)]))) 
        self.mood_vocab = sorted(list(set([t for tags in self.df["MOOD_TAGS"] for t in ast.literal_eval(tags)])))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        song_id = os.path.splitext(row["PATH"])[0]  # Remove the file extension to get the song ID

        emb = self.audio_cache[song_id]  # (N, emb_size)
        
        # Build multi-hot tag vectors
        g_tags = ast.literal_eval(row["GENRE_TAGS"])
        i_tags = ast.literal_eval(row["INSTRUMENT_TAGS"])
        m_tags = ast.literal_eval(row["MOOD_TAGS"])

        g_vec = torch.tensor([1.0 if t in g_tags else 0.0 for t in self.genre_vocab])
        i_vec = torch.tensor([1.0 if t in i_tags else 0.0 for t in self.inst_vocab])
        m_vec = torch.tensor([1.0 if t in m_tags else 0.0 for t in self.mood_vocab])

        artist_id = torch.tensor(self.artist_to_idx[str(row["ARTIST_ID"])], dtype=torch.long)

        return emb, i_vec, g_vec, m_vec, artist_id


class TripletDatasetVal(TripletDataset):
    def __init__(self, backbone, DATASETS_FOLDER, pooling):
        self.pooling = pooling

        self.triplets_df = pd.read_csv(f"{DATASETS_FOLDER}/jamendo/jamendo_val_triplets.csv")
        #get a debug sample of 100
        self.triplets_df = self.triplets_df.sample(n=100, random_state=42).reset_index(drop=True)

        all_songs = pd.concat([self.triplets_df["anchor"], self.triplets_df["positive"], self.triplets_df["negative"]]).unique()

        super().__init__(backbone, DATASETS_FOLDER, all_songs, "val", pooling)

    def __len__(self):
        return len(self.triplets_df)

    def __getitem__(self, idx):
        row = self.triplets_df.iloc[idx]

        anchor = os.path.splitext(row['anchor'])[0]
        negative = os.path.splitext(row['negative'])[0]
        positive = os.path.splitext(row['positive'])[0]
        
        emb_anchor = self.audio_cache[anchor]
        emb_negative = self.audio_cache[negative]
        emb_positive = self.audio_cache[positive]

        return emb_anchor, emb_negative, emb_positive, self.condition_mapping[row['category']]