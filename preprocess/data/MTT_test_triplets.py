import pandas as pd
import numpy as np
import os
import json
import random
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
import re


tag_categories = {
    'instrument': ['no voice', 'singer', 'duet', 'plucking','bongos', 'harpsichord', 'female singing',
                    'sitar', 'chorus', 'female opera', 'male vocal', 'vocals', 'clarinet', 'beats', 'men', 
                    'woodwind', 'no strings', 'chimes', 'no piano', 'horns', 'female', 'no voices',
                    'guitar', 'quiet', 'no beat', 'banjo', 'electric', 'violins', 'female voice', 'wind', 'synth',
                    'no singing', 'trumpet', 'percussion', 'drum', 'voice', 'birds', 'strings', 'bass', 'harpsicord', 
                    'male voice', 'girl', 'keyboard', 'string', 'drums', 'chanting', 'no violin',  'no guitar', 
                    'organ', 'no vocal', 'talking', 'choral',  'soprano',  'acoustic guitar', 'electric guitar', 
                    'male singer', 'man singing', 'classical guitar', 'violin', 'no vocals', 'horn', 'instrumental',
                    'chant', 'synthesizer', 'bells', 'man', 'harp', 'no flute', 'female vocal', 'oboe', 
                    'viola','piano', 'male vocals', 'flutes', 'sax', 'vocal','no singer', 'male', 'woman', 'woman singing', 
                    'piano solo', 'guitars', 'no drums', 'singing', 'cello', 'female vocals', 'voices', 'clapping',
                    'flute', 'beat', 'women', 'fiddle', 'choir', 'female singer','water', 'sax', 'vocal', 'lute'], 
    
    'genre': ['hard rock', 'world', 'clasical', 'heavy', 'silence', 'funky', 'foreign', 'classical', 
                'soft rock', 'solo', 'jazz', 'folk', 'ambient', 'new age', 'funk', 'middle eastern',  
                'medieval', 'acoustic', 'classic', 'electronic', 'not classical', 'opera', 'not rock', 
                'country', 'electro', 'reggae', 'tribal', 'male opera', 'irish', 'electronica', 'operatic', 
                'arabic',  'trance', 'drone', 'heavy metal', 'modern', 'disco', 'industrial', 'jungle', 
                'pop', 'celtic', 'eastern', 'old', 'punk', 'spanish', 'english','orchestral', 'orchestra',
                'blues','indian', 'india', 'rock', 'dance',  'jazzy', 'techno', 'house', 'monks','rap', 'metal', 
                'hip hop', 'not english', 'oriental', 'baroque', 'not opera',], 
    
    'mood': ['eerie', 'spacey', 'happy', 'airy', 'weird', 'loud', 'space', 'repetitive', 'fast', 'dark', 'lol', 'low',
            'strange', 'deep', 'fast beat', 'hard', 'mellow', 'light', 'echo', 'sad', 'scary', 'slow', 'calm', 
            'different', 'upbeat', 'soft', 'noise', 'quick', 'electric'],
}

clean_tags = {
    'no voice': 'no voice',
    'singer': 'vocals',
    'duet': 'vocals',
    'plucking': 'plucking',
    'hard rock': 'rock',
    'world': 'foreign',
    'bongos': 'drumming',
    'harpsichord': 'harpsichord',
    'female singing': 'vocals',
    'clasical': 'classical',
    'sitar': 'sitar',
    'chorus': 'vocals',
    'female opera': 'opera',
    'male vocal': 'vocals',
    'vocals': 'vocals',
    'clarinet': 'woodwind',
    'heavy': 'heavy',
    'silence': 'silence',
    'beats': 'drumming',
    'men': 'vocals',
    'woodwind': 'woodwind',
    'funky': 'funky',
    'no strings': 'no strings',
    'chimes': 'chimes',
    'foreign': 'foreign',
    'no piano': 'no piano',
    'horns': 'horns',
    'classical': 'classical',
    'female': 'vocals',
    'no voices': 'no voice',
    'soft rock': 'rock',
    'eerie': 'eerie',
    'spacey': 'spacey',
    'jazz': 'jazz',
    'guitar': 'guitar',
    'quiet': 'quiet',
    'no beat': 'no beat',
    'banjo': 'banjo',
    'electric': 'electric',
    'solo': 'solo',
    'violins': 'strings',
    'folk': 'folk',
    'female voice': 'vocals',
    'wind': 'wind',
    'happy': 'happy',
    'ambient': 'ambient',
    'new age': 'new age',
    'synth': 'synth',
    'funk': 'funk',
    'no singing': 'no voice',
    'middle eastern': 'middle eastern',
    'trumpet': 'trumpet',
    'percussion': 'drumming',
    'drum': 'drumming',
    'airy': 'airy',
    'voice': 'vocals',
    'repetitive': 'repetitive',
    'birds': 'birds',
    'space': 'spacey',
    'strings': 'strings',
    'bass': 'bass',
    'harpsicord': 'harpsichord',
    'medieval': 'medieval',
    'male voice': 'vocals',
    'girl': 'vocals',
    'keyboard': 'piano',
    'acoustic': 'acoustic',
    'loud': 'loud',
    'classic': 'classical',
    'string': 'strings',
    'drums': 'drumming',
    'electronic': 'electronic',
    'not classical': 'not classical',
    'chanting': 'chanting',
    'no violin': 'no strings',
    'not rock': 'not rock',
    'no guitar': 'no guitar',
    'organ': 'piano',
    'no vocal': 'no voice',
    'talking': 'vocals',
    'choral': 'vocals',
    'weird': 'weird',
    'opera': 'opera',
    'soprano': 'vocals',
    'fast': 'fast',
    'acoustic guitar': 'acoustic guitar',
    'electric guitar': 'electric guitar',
    'male singer': 'vocals',
    'man singing': 'vocals',
    'classical guitar': 'acoustic guitar',
    'country': 'country',
    'violin': 'violin',
    'electro': 'electronic',
    'reggae': 'reggae',
    'tribal': 'tribal',
    'dark': 'dark',
    'male opera': 'opera',
    'no vocals': 'no voice',
    'irish': 'irish',
    'electronica': 'electronic',
    'horn': 'horns',
    'operatic': 'opera',
    'arabic': 'arabic',
    'lol': 'weird',
    'low': 'low',
    'instrumental': 'no voice',
    'trance': 'electronic',
    'chant': 'chanting',
    'strange': 'weird',
    'drone': 'drone',
    'synthesizer': 'synth',
    'heavy metal': 'heavy metal',
    'modern': 'modern',
    'disco': 'disco',
    'bells': 'bells',
    'man': 'vocals',
    'deep': 'deep',
    'fast beat': 'fast beat',
    'industrial': 'industrial',
    'hard': 'hard',
    'harp': 'harp',
    'no flute': 'no flute', 
    'jungle': 'jungle',
    'pop': 'pop',
    'lute': 'medieval',
    'female vocal': 'vocals',
    'oboe': 'woodwind',
    'mellow': 'mellow',
    'orchestral': 'orchestral',
    'viola': 'strings',
    'light': 'light',
    'echo': 'echo',
    'piano': 'piano',
    'celtic': 'celtic',
    'male vocals': 'vocals',
    'orchestra': 'orchestral',
    'eastern': 'eastern',
    'old': 'old',
    'flutes': 'woodwind',
    'punk': 'punk',
    'spanish': 'spanish',
    'sad': 'sad',
    'sax': 'sax',
    'slow': 'slow',
    'male': 'vocals',
    'blues': 'blues',
    'vocal': 'vocals',
    'indian': 'indian',
    'no singer': 'no voice',
    'scary': 'scary',
    'india': 'indian',
    'woman': 'vocals',
    'woman singing': 'vocals',
    'rock': 'rock',
    'dance': 'dance',
    'piano solo': 'piano',
    'guitars': 'guitars',
    'no drums': 'no drums',
    'jazzy': 'jazz',
    'singing': 'vocals',
    'cello': 'strings',
    'calm': 'calm',
    'female vocals': 'vocals',
    'voices': 'vocals',
    'different': 'different',
    'techno': 'techno',
    'clapping': 'clapping',
    'house': 'house',
    'monks': 'monks',
    'flute': 'woodwind',
    'not opera': 'not opera',
    'not english': 'not english',
    'oriental': 'oriental',
    'beat': 'drumming',
    'upbeat': 'upbeat',
    'soft': 'soft',
    'noise': 'noise',
    'choir': 'chanting',
    'female singer': 'vocals',
    'rap': 'hip hop',
    'metal': 'heavy metal',
    'hip hop': 'hip hop',
    'quick': 'fast',
    'water': 'water',
    'baroque': 'classical',
    'women': 'vocals',
    'fiddle': 'strings',
    'english': 'english',
}

def clean_magnatagatune_tags(df, clean_tags):
    """
    Clean the tags in the MagnaTagATune dataset based on a provided mapping.

    Parameters:
    df (pd.DataFrame): The original DataFrame containing the tags.
    clean_tags (dict): A dictionary mapping original tags to cleaned tags.

    Returns:
    pd.DataFrame: A new DataFrame with cleaned tags.
    """
    # 1. Clean whitespace from your dictionary values
    clean_tags_sanitized = {k: v.strip() for k, v in clean_tags.items()}

    # 2. Identify metadata vs tag columns
    meta_cols = [c for c in ["clip_id", "mp3_path"] if c in df.columns]

    # 3. Group original columns by their clean target tag
    target_to_orig_cols = defaultdict(list)
    for orig_col, clean_col in clean_tags_sanitized.items():
        if orig_col in df.columns:
            target_to_orig_cols[clean_col].append(orig_col)

    # 4. Aggregate binary columns (1 if any original tag is 1, else 0)
    clean_tag_dfs = {}
    for clean_col, orig_cols in target_to_orig_cols.items():
        clean_tag_dfs[clean_col] = df[orig_cols].max(axis=1)

    df_clean_tags = pd.DataFrame(clean_tag_dfs, index=df.index)

    # 5. Re-combine metadata with clean tags
    df_clean = pd.concat([df[meta_cols], df_clean_tags], axis=1)

    # MagnaTagATune paths end with timestamps like '-30-59.mp3'
    # This regex strips the timestamp to isolate the unique song identifier
    df_clean["song_id"] = df_clean["mp3_path"].apply(
        lambda path: re.sub(r"-\d+-\d+\.mp3$", "", str(path))
    )

    return df_clean

def clean_tag_categories(tag_categories, clean_tags):
    tag_categories_clean = {'instrument': [], 'genre': [], 'mood': []}
    tag_categories_clean['instrument'] = list(set([clean_tags[tag] for tag in tag_categories['instrument']]))
    tag_categories_clean['genre'] = list(set([clean_tags[tag] for tag in tag_categories['genre']]))
    tag_categories_clean['mood'] = list(set([clean_tags[tag] for tag in tag_categories['mood']]))

    return tag_categories_clean

def generate_song_clean_triplets(df_clean, clip_info_df, category_name, category_tags, num_triplets=5000, seed=42):

    random.seed(seed)

    #keep only rows that have at least one of the tags in the category
    valid_tags = [tag for tag in category_tags if tag in df_clean.columns]
    cat_df = df_clean[df_clean[valid_tags].sum(axis=1) > 0].copy()
    cat_df = cat_df.set_index("mp3_path")

    artist_map = clip_info_df.set_index("mp3_path")["artist"].to_dict()
    cat_df["artist"] = cat_df.index.map(artist_map)

    all_clip_ids = cat_df.index.tolist()
    triplets = []
    max_attempts = 50
    attempts = 0

    while len(triplets) < num_triplets and attempts < max_attempts:
        print(f"Generating triplets for '{category_name}': {len(triplets)}/{num_triplets}. Attempts: {attempts}", end='\r')
        attempts += 1

        # 1. Select Anchor
        anchor_id = random.choice(all_clip_ids)
        anchor_row = cat_df.loc[anchor_id]
        anchor_artist = anchor_row["artist"]

        active_anchor_tags = anchor_row[valid_tags]
        active_anchor_tags = active_anchor_tags[active_anchor_tags == 1].index.tolist()

        if not active_anchor_tags:
            continue

        # 2. Select Positive: shares tag
        different_from_anchor_mask = cat_df["artist"] != anchor_artist
        pos_tag_mask = (cat_df[active_anchor_tags] == 1).any(axis=1)
        pos_candidates = cat_df[pos_tag_mask & different_from_anchor_mask].index

        if len(pos_candidates) == 0:
            continue

        pos_id = random.choice(pos_candidates.tolist())
        pos_artist = cat_df.loc[pos_id, "artist"]

        # 3. Select Negative: shares NO tag AND comes from a DIFFERENT artist than BOTH Anchor and Positive
        different_from_both_mask = (cat_df["artist"] != anchor_artist) & (cat_df["artist"] != pos_artist)
        neg_tag_mask = (cat_df[active_anchor_tags] == 0).all(axis=1)
        neg_candidates = cat_df[neg_tag_mask & different_from_both_mask].index

        if len(neg_candidates) == 0:
            continue

        neg_id = random.choice(neg_candidates.tolist())

        triplets.append(
            {
                "anchor": anchor_id,
                "positive": pos_id,
                "negative": neg_id,
                "category": category_name,
            }
        )

        attempts = 0  # Reset attempt counter on successful generation

    print(f"Generated {len(triplets)} triplets for '{category_name}'")

    return triplets

def main():
    random.seed(42)

    load_dotenv()
    DATASETS_FOLDER = os.environ["DATASETS_FOLDER"]
    df = pd.read_csv(f'{DATASETS_FOLDER}/magnatagatune/annotations_final.csv', sep='\t')

    corrupted_files = ["9/american_baroque-dances_and_suites_of_rameau_and_couperin-25-le_petit_rien_xiveme_ordre_couperin-88-117.mp3", 
                        "8/jacob_heringman-josquin_des_prez_lute_settings-19-gintzler__pater_noster-204-233.mp3",
                        "6/norine_braun-now_and_zen-08-gently-117-146.mp3"]

    #drop corrupted files
    df = df[~df['mp3_path'].isin(corrupted_files)].copy()

    clip_info_df = pd.read_csv(f'{DATASETS_FOLDER}/magnatagatune/clip_info_final.csv', sep="\t")
    
    df_clean = clean_magnatagatune_tags(df, clean_tags)
    tag_categories_clean = clean_tag_categories(tag_categories, clean_tags)

    # Generate across all cleaned categories
    test_benchmark = []
    for category, tags in tag_categories_clean.items():
        print(f'generating {category} triplets')
        test_benchmark.extend(generate_song_clean_triplets(df_clean, clip_info_df, category, tags, num_triplets=5000))

    test_benchmark_df = pd.DataFrame(test_benchmark)
    test_benchmark_df.to_csv(f'{DATASETS_FOLDER}/magnatagatune/test_triplets.csv', index=False)

if __name__ == "__main__":
    main()

