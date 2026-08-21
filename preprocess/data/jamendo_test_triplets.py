import pandas as pd
import numpy as np
import os
import json
import random
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
import re
import ast

clean_name = {
    'GENRE_TAGS': 'genre',
    'INSTRUMENT_TAGS': 'instrument',
    'MOOD_TAGS': 'mood'
}

def generate_triplets(df, category_name, num_triplets=5000):

    # Keep tracks containing at least one tag in this category
    cat_df = df[df[category_name].apply(lambda tags: len(tags) > 0)].copy()
    cat_df["tag_set"] = cat_df[category_name].apply(set)

    if "PATH" in cat_df.columns:
        cat_df = cat_df.set_index("PATH")

    all_clip_ids = cat_df.index.tolist()
    triplets = []
    max_attempts = 50
    attempts = 0

    # Extract Series for vectorized filtering
    artists = cat_df["ARTIST_ID"]
    tag_sets = cat_df["tag_set"]

    while len(triplets) < num_triplets and attempts < max_attempts:
        print(f"Generating triplets for '{category_name}': {len(triplets)}/{num_triplets}. Attempts: {attempts}",end="\r",)
        attempts += 1

        # 1. Select Anchor
        anchor_id = random.choice(all_clip_ids)
        anchor_artist = artists.loc[anchor_id]
        anchor_tags = tag_sets.loc[anchor_id]

        # 2. Select Positive: shares >=1 tag
        shares_tag = tag_sets.apply(lambda s: not s.isdisjoint(anchor_tags))
        pos_candidates = [cid for cid in cat_df[shares_tag].index.tolist() if cid != anchor_id]
        
        if not pos_candidates:
            continue

        pos_id = random.choice(pos_candidates)
        pos_artist = artists.loc[pos_id]

        # 3. Select Negative: shares 0 tags AND comes from a DIFFERENT artist than BOTH Anchor and Positive
        diff_both_artists = (artists != anchor_artist) & (artists != pos_artist)
        no_shared_tag = tag_sets.apply(lambda s: s.isdisjoint(anchor_tags))
        neg_candidates = cat_df[diff_both_artists & no_shared_tag].index.tolist()

        if not neg_candidates:
            continue

        neg_id = random.choice(neg_candidates)

        triplets.append(
            {
                "anchor": anchor_id,
                "positive": pos_id,
                "negative": neg_id,
                "category": clean_name[category_name],
            }
        )

        attempts = 0  # Reset attempt counter on success

    print(f"\nGenerated {len(triplets)} triplets for '{category_name}'")

    return triplets

def triplets_from_df(df, save_path):
    # Ensure TAGS are Python lists
    if isinstance(df["GENRE_TAGS"].iloc[0], str):
        df["GENRE_TAGS"] = df["GENRE_TAGS"].apply(ast.literal_eval)
    if isinstance(df["INSTRUMENT_TAGS"].iloc[0], str):
        df["INSTRUMENT_TAGS"] = df["INSTRUMENT_TAGS"].apply(ast.literal_eval)
    if isinstance(df["MOOD_TAGS"].iloc[0], str):
        df["MOOD_TAGS"] = df["MOOD_TAGS"].apply(ast.literal_eval)

    test_benchmark = []
    tag_categories = ['INSTRUMENT_TAGS', 'MOOD_TAGS', 'GENRE_TAGS']

    for category in tag_categories:
        print(f'generating {category} triplets')
        test_benchmark.extend(generate_triplets(df, category, num_triplets=5000))

    test_benchmark_df = pd.DataFrame(test_benchmark)
    test_benchmark_df.to_csv(save_path, index=False) 

def main():
    load_dotenv()
    DATASETS_FOLDER = os.environ["DATASETS_FOLDER"]

    random.seed(42)
    
    df_test = pd.read_csv(f"{DATASETS_FOLDER}/jamendo/jamendo_test_split.csv")
    df_val = pd.read_csv(f"{DATASETS_FOLDER}/jamendo/jamendo_val_split.csv")

    triplets_from_df(df_test, f"{DATASETS_FOLDER}/jamendo/jamendo_test_triplets.csv")
    triplets_from_df(df_val, f"{DATASETS_FOLDER}/jamendo/jamendo_val_triplets.csv")

if __name__ == "__main__":
    main()