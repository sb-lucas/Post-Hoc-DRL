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

def generate_triplets(df, category_name, num_triplets=5000, seed=42):
    random.seed(seed)

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

        # 2. Select Positive: shares >=1 tag AND comes from a DIFFERENT artist
        diff_anchor_artist = artists != anchor_artist
        shares_tag = tag_sets.apply(lambda s: not s.isdisjoint(anchor_tags))
        pos_candidates = cat_df[diff_anchor_artist & shares_tag].index.tolist()

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
                "category": category_name,
            }
        )

        attempts = 0  # Reset attempt counter on success

    print(f"\nGenerated {len(triplets)} triplets for '{category_name}'")

    return triplets

def main():
    df = pd.read_csv("jamendo_test_split.csv")

    # Ensure TAGS are Python lists
    if isinstance(df["TAGS"].iloc[0], str):
        df["TAGS"] = df["TAGS"].apply(ast.literal_eval)

    # Populate category columns for each row
    df["genre"] = df["TAGS"].apply(lambda tags: [tag for tag in tags if tag.startswith("genre---")])
    df["instrument"] = df["TAGS"].apply(lambda tags: [tag for tag in tags if tag.startswith("instrument---")])
    df["mood"] = df["TAGS"].apply(lambda tags: [tag for tag in tags if tag.startswith("mood/theme---")])

    test_benchmark = []
    tag_categories = ['instrument', 'mood', 'genre']

    for category in tag_categories:
        print(f'generating {category} triplets')
        test_benchmark.extend(generate_triplets(df, category, num_triplets=5000))

    test_benchmark_df = pd.DataFrame(test_benchmark)
    test_benchmark_df.to_csv("jamendo_test_triplets.csv", index=False) 

if __name__ == "__main__":
    main()