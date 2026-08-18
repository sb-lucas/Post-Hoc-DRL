import pandas as pd
import numpy as np
import os
import json
import random
from collections import defaultdict
from dotenv import load_dotenv
import re
import ast
from sklearn.model_selection import GroupShuffleSplit

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
        "TAGS": row[5:]
    })

df = pd.DataFrame(data)

# Ensure TAGS are Python lists
if isinstance(df["TAGS"].iloc[0], str):
    df["TAGS"] = df["TAGS"].apply(ast.literal_eval)

# Group-based split (80% train / 20% test) by ARTIST_ID
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df["ARTIST_ID"]))

df_train = df.iloc[train_idx].copy().reset_index(drop=True)
df_test = df.iloc[test_idx].copy().reset_index(drop=True)

# Save train tracks for future online semi-hard triplet loader
df_train[["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "TAGS"]].to_csv("jamendo_train_split.csv", index=False)
df_test[["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "TAGS"]].to_csv("jamendo_test_split.csv", index=False)

print(f"Train tracks: {len(df_train)} | Test tracks: {len(df_test)} (Zero artist overlap)")