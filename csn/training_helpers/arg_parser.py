import argparse
from email import parser

#nohup python3 -u main.py --backbone clap --variation none --learn_masks --epochs 100 > clap.out & 

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument('--n_conditions', type=int, default=3, help='Number of Conditions')

    parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--epochs', type=int, required=True, help='Number of Training Epochs')

    parser.add_argument('--backbone', type=str, required=True, help="Backbone Model Name: 'mert', 'clap' or 'muq'")
    parser.add_argument('--variation', type=str, required=True, help="Additional Weights for Finetuning Variation: 'one_layer', 'mlp' or 'none'")
    parser.add_argument('--pooling', type=str, required=True, help="Pooling Method: 'mean' or 'attention'")
    parser.add_argument('--learn_masks', action='store_true', help='Whether to learn masks')

    args = parser.parse_args()

    return args