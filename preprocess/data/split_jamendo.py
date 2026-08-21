import pandas as pd
import numpy as np
import os
import json
import random
from collections import defaultdict
import re
import ast
from sklearn.model_selection import GroupShuffleSplit
from dotenv import load_dotenv

load_dotenv()
DATASETS_FOLDER = os.environ["DATASETS_FOLDER"]

path = f"{DATASETS_FOLDER}/mtg-jamendo-dataset/data/raw_30s_cleantags.tsv"

with open(path, "r", encoding="utf-8") as f:
    rows = [line.rstrip("\n").split("\t") for line in f]

# First row is the header
header = rows[0]

# First 5 columns are fixed metadata
fixed_columns = header[:5]

# Everything after that is a tag
data = []

for row in rows[1:]:
    data.append({
        "TRACK_ID": row[0],
        "ARTIST_ID": row[1],
        "ALBUM_ID": row[2],
        "PATH": row[3],
        "DURATION": float(row[4]),
        "GENRE_TAGS": [t for t in row[5:] if t.startswith("genre")],
        "INSTRUMENT_TAGS": [t for t in row[5:] if t.startswith("instrument")],
        "MOOD_TAGS": [t for t in row[5:] if t.startswith("mood")],
    })

df = pd.DataFrame(data)

# Ensure TAGS are Python lists
if isinstance(df["GENRE_TAGS"].iloc[0], str):
    df["GENRE_TAGS"] = df["GENRE_TAGS"].apply(ast.literal_eval)
if isinstance(df["INSTRUMENT_TAGS"].iloc[0], str):
    df["INSTRUMENT_TAGS"] = df["INSTRUMENT_TAGS"].apply(ast.literal_eval)
if isinstance(df["MOOD_TAGS"].iloc[0], str):
    df["MOOD_TAGS"] = df["MOOD_TAGS"].apply(ast.literal_eval)

# First split: 80% train / 20% temporary (val + test) by ARTIST_ID
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, temp_idx = next(gss.split(df, groups=df["ARTIST_ID"]))

df_train = df.iloc[train_idx].copy().reset_index(drop=True)
df_temp = df.iloc[temp_idx].copy().reset_index(drop=True)

# Second split: split the 20% temporary set equally into 10% val / 10% test
gss_temp = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(
    gss_temp.split(df_temp, groups=df_temp["ARTIST_ID"])
)

df_val = df_temp.iloc[val_idx].copy().reset_index(drop=True)
df_test = df_temp.iloc[test_idx].copy().reset_index(drop=True)

# Save train tracks for future online semi-hard triplet loader
columns_to_save = ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "GENRE_TAGS", "INSTRUMENT_TAGS", "MOOD_TAGS"]
df_train[columns_to_save].to_csv(f"{DATASETS_FOLDER}/jamendo/jamendo_train_split.csv", index=False)
df_val[columns_to_save].to_csv(f"{DATASETS_FOLDER}/jamendo/jamendo_val_split.csv", index=False)
df_test[columns_to_save].to_csv(f"{DATASETS_FOLDER}/jamendo/jamendo_test_split.csv", index=False)

print(f"Train tracks: {len(df_train)} | Val tracks: {len(df_val)} | Test tracks: {len(df_test)} (Zero artist overlap)")