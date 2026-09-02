# Exploring Post-Hoc Disentanglement of Audio Foundation Models for Music Similarity

Official repository for the 2026 LAMIR paper.

## Description
This work investigates post-hoc disentanglement of pretrained audio foundation models using Conditional Similarity Networks (CSN). We adapt CSN to gate frozen CLAP, MuQ, and MERT embeddings, evaluating lightweight adaptation and masking strategies across two music similarity datasets.

## Structure

```text
DRL-Music-Post-Hoc/
├── csn/
│   ├── training_helpers/     # Helper code for training, including logging, dataloaders, semi-hard triplet loss, and others
│   ├── csn.py                # Conditional Similarity Network class
│   ├── variations.py         # Adaptation module classes
│   ├── main.py               # Main script for training the models
│   ├── evaluate.py           # Script to evaluate triplet score performance of all trained configurations
│   └── generate_tables.py    # Script to reproduce the values reported in the paper's tables
├── preprocess/
│   ├── data/                 # Scripts for cleaning, splitting, and creating triplets from the MagnaTagATune and MTG-Jamendo datasets
│   └── features/             # Scripts for extracting and downsampling features from the pretrained audio foundation models
├── .gitignore
├── README.md                 # Project documentation (this file)
└── requirements.txt          # Dependencies file for reproducibility
```

## How to Start

### Installing Requirements

The code was developed and tested with **Python 3.11.15**.

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

### Reproducing Results

To reproduce the results, first download the data.

For MTG-Jamendo, check: https://mtg.github.io/mtg-jamendo-dataset/

For MagnaTagATune, check: https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset

Preprocess the data using the scripts in `preprocess/`. Set up your `.env` file with the corresponding paths on your machine.

For training, the main training script is `csn/main.py`. Run it with the following arguments:

| Argument         | Description                                         |      Default |
| ---------------- | --------------------------------------------------- | -----------: |
| `--n_conditions` | Number of conditions                                |          `3` |
| `--lr`           | Learning rate                                       |       `1e-4` |
| `--batch_size`   | Training batch size                                 |        `128` |
| `--epochs`       | Number of training epochs                           | **Required** |
| `--backbone`     | Backbone model: `mert`, `clap`, or `muq`            | **Required** |
| `--variation`    | Finetuning variation: `one_layer`, `mlp`, or `none` | **Required** |
| `--pooling`      | Pooling method: `mean` or `attention`               | **Required** |
| `--learn_masks`  | Enable learning of masks                            |      `False` |

Finally, run `evaluate.py` to evaluate the trained models on the two test sets. Then, run `generate_tables.py` to reproduce the tables from the paper.
