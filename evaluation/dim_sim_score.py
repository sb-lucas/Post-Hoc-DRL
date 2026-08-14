import os
import random
import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import argparse
import librosa
import pandas as pd
import os
import librosa
from transformers import AutoProcessor, ClapModel
import torch
import torch.nn.functional as F
from essentia.standard import MonoLoader, TensorflowPredictMusiCNN
import numpy as np
import gc

WEIGHTS_PATH = '/home/ldsbrandao/exp/diego/diego/final/essentia_model_weights'

#function to calculate how well each model performs in the dim sim dataset
class CLAP():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

    def get_audio_embeddings(self, audio, sr):
        #resample to 48kHz if needed
        if sr != 48000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)

        audio_data = audio.reshape(1, -1)  # (1, T)

        input_text = [""]
        inputs = self.processor(text=input_text, audios=audio_data, return_tensors="pt", 
                                padding=True, sampling_rate=48000).to(self.device)

        outputs = self.model(**inputs)

        return outputs.audio_embeds

class MusiCNN():
    def __init__(self, ):
        self.model = TensorflowPredictMusiCNN(graphFilename=os.path.join(WEIGHTS_PATH, "msd-musicnn-1.pb"), 
                                              output="model/dense/BiasAdd")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def get_audio_embeddings(self, audio, sr):
        # resample to 16kHz if needed
        target_sr = 16000
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        embeddings = self.model(audio)

        #take the mean over time and convert to tensor
        embeddings = torch.tensor(embeddings).mean(dim=0, keepdim=True)
        embeddings = embeddings.to(self.device)

        return embeddings  

def load_audio(song_id, start_samples, clip_length_samples, sr):
    file_path = f'/home/ldsbrandao/exp/diego/diego/final/dimsim/clean_dimsim_audios/{song_id}.mp3'
    audio, _ = librosa.load(file_path, sr=sr, offset=start_samples/sr, duration=clip_length_samples/sr)
    return audio

def solve_triplet(model, sr, anchor, song1, song2, mask):
    #extract embeddings with model
    anchor_emb = model.get_audio_embeddings(anchor, sr)
    song1_emb = model.get_audio_embeddings(song1, sr)
    song2_emb = model.get_audio_embeddings(song2, sr)

    #ensure normalized embeddings
    anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
    song1_emb = F.normalize(song1_emb, p=2, dim=1)
    song2_emb = F.normalize(song2_emb, p=2, dim=1)

    if mask is not None:
        anchor_emb = anchor_emb * mask
        song1_emb = song1_emb * mask
        song2_emb = song2_emb * mask

    #calculate cosine similarity
    sim_song1 = F.cosine_similarity(anchor_emb, song1_emb)
    sim_song2 = F.cosine_similarity(anchor_emb, song2_emb)

    #return 1 if song1 is more similar to the anchor, else return 2
    if sim_song1 > sim_song2:
        return 1
    else:        
        return 2


def evaluate_dimsim(model_name, masks):
    if model_name == "clap":
        model = CLAP()
    elif model_name == "musicnn":
        model = MusiCNN()

    df = pd.read_csv('/home/ldsbrandao/exp/diego/diego/final/dimsim/metadata/clean-dim-sim.csv')

    counter = {"backbone": 0, "instrument": 0, "genre": 0, "theme": 0}
    condition_mapping = {
        'instrument':0,
        'genre':1,
        'theme':2
    }

    for index, row in df.iterrows():
        print(f"Evaluating DimSim - Sample {index+1}/{len(df)}", end='\r')

        sr = row['sampling_rate']

        anchor_audio = load_audio(row['anchor_id'], row['anchor_start_samples'], row['clip_lengths_samples'], sr)
        song1_audio = load_audio(row['song1_id'], row['song1_start_samples'], row['clip_lengths_samples'], sr)
        song2_audio = load_audio(row['song2_id'], row['song2_start_samples'], row['clip_lengths_samples'], sr)

        for condition in counter.keys():
            #mask the audio according to the condition, if not backbone
            if condition != 'backbone':
                mask = masks[condition_mapping[condition]]
            else:
                mask = None

            model_sim = solve_triplet(model, sr, anchor_audio, song1_audio, song2_audio, mask)
            gt_sim = 1 if row['song1_vote'] == 1 else 2

            if model_sim == gt_sim:
                counter[condition] += 1

    acc = {condition: counter[condition] / len(df) for condition in counter.keys()}

    # cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return acc

    