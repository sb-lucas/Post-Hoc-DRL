import os
import numpy as np
import pandas as pd
import random
import re
import ast
import sys
from collections import defaultdict

def split_train_test(df, train_ratio=0.8, val_ratio=0.1):
    # Get list of unique songs
    songs = df['song'].unique().tolist()

    # Shuffle
    random.seed(42)  # for reproducibility
    random.shuffle(songs)

    # Calculate index cutoffs
    num_songs = len(songs)
    train_end = int(num_songs * train_ratio)
    val_end = train_end + int(num_songs * val_ratio)

    # Slice the unique songs into their respective sets
    train_songs = songs[:train_end]
    val_songs = songs[train_end:val_end]
    test_songs = songs[val_end:] # Everything left over becomes test (approx 1-train_ratio-val_ratio)

    # Create a DataFrame to store the splits
    splits_df = pd.DataFrame({
        'song': songs,
        'split': ['train' if song in train_songs else 'val' if song in val_songs else 'test' for song in songs]
    })

    #need to make sure all modification types are present in all splits

    return splits_df

def main():

    #load tag information
    df_multilabel = pd.read_csv("/home/ldsbrandao/exp/cerri/dataset/song_labels_multilabel.csv", index_col=0)
    df_genre = pd.read_csv("/home/ldsbrandao/exp/cerri/dataset/song_genre.csv")
    df_genre["genre"] = df_genre["genre"].apply(lambda x: [x])

    #split train and test songs
    splits_df = split_train_test(df_genre).set_index('song')

    #-----------------------------Match songs and Tags-----------------------------#
    # All tag columns
    tag_cols = [c for c in df_multilabel.columns if c != 'song' and c != 'multilabel']
    instrument_cols = [c for c in tag_cols if re.match(r'^instrument_', c)]
    theme_cols = [c for c in tag_cols if re.match(r'^theme_', c)]

    #select tags for each song based on a threshold (0.5 for instruments, 0.3 for themes)
    df_multilabel["instrument"] = (df_multilabel[instrument_cols].gt(0.5).apply(lambda row: row.index[row].tolist(), axis=1))
    df_multilabel["theme"] = (df_multilabel[theme_cols].gt(0.3).apply(lambda row: row.index[row].tolist(), axis=1))

    #merge the splits with the tags and genre information
    splits_df = splits_df.join(df_multilabel.set_index("song")["instrument"], on="song")    
    splits_df = splits_df.join(df_multilabel.set_index("song")["theme"], on="song")
    splits_df = splits_df.join(df_genre.set_index("song")["genre"], on="song")

    #-----------------------------Create Triplets-----------------------------#
    triplets_rows = {}
    for split in ['train', 'val', 'test']:
        df = splits_df[splits_df['split'] == split]
        all_songs = set(df.index)

        # Create a mapping from each tag to the songs that have that tag for each condition
        tag_to_songs = {}
        for condition in ['instrument', 'theme', 'genre']:
            mapping = defaultdict(set)
            
            for idx, tags in zip(df.index, df[condition]):
                for tag in tags:
                    mapping[tag].add(idx)

            tag_to_songs[condition] = mapping

        triplets_rows[split] = []

        #create triplets for each song in the split
        for cont, (idx, row) in enumerate(df.iterrows()):
            print(f"Processing {split} song {cont+1}/{len(df)}", end='\r')

            for condition in ['instrument', 'theme', 'genre']:
                anchor_tags = row[condition]

                if not anchor_tags:
                    continue

                # positive songs are those that share at least one tag with the anchor song
                positive_songs = set()
                for tag in anchor_tags:
                    positive_songs.update(tag_to_songs[condition][tag])
                positive_songs.discard(idx)

                negative_songs = list(all_songs - positive_songs - {idx})

                if not negative_songs:
                    continue

                for pos in random.sample(list(positive_songs), min(len(positive_songs), 5)):
                    triplets_rows[split].append({
                        "anchor": idx,
                        "positive": pos,
                        "negative": random.choice(negative_songs),
                        "condition": condition
                    })

    #save triplets to csv
    train_triplets_df = pd.DataFrame(triplets_rows['train']).to_csv('train_triplets.csv', index=False)
    val_triplets_df = pd.DataFrame(triplets_rows['val']).to_csv('val_triplets.csv', index=False)
    test_triplets_df = pd.DataFrame(triplets_rows['test']).to_csv('test_triplets.csv', index=False)

if __name__ == "__main__":
    main()