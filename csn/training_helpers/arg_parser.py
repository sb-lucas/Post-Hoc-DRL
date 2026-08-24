import argparse
from email import parser


#nohup python3 -u main.py --epochs 1000 --backbone clap --variation none --pooling attention --learn_masks
#nohup python3 -u main.py --epochs 1000 --backbone clap --variation none --pooling mean --learn_masks
#nohup python3 -u main.py --epochs 1000 --backbone clap --variation one_layer --pooling attention --learn_masks
#nohup python3 -u main.py --epochs 1000 --backbone clap --variation one_layer --pooling attention 
#nohup python3 -u main.py --epochs 1000 --backbone clap --variation one_layer --pooling mean --learn_masks
#nohup python3 -u main.py --epochs 1000 --backbone clap --variation one_layer --pooling mean
#nohup python3 -u main.py --epochs 1000 --backbone clap --variation mlp --pooling attention --learn_masks
#nohup python3 -u main.py --epochs 1000 --backbone clap --variation mlp --pooling attention 
#nohup python3 -u main.py --epochs 1000 --backbone clap --variation mlp --pooling mean --learn_masks
#nohup python3 -u main.py --epochs 1000 --backbone clap --variation mlp --pooling mean 

#nohup python3 -u main.py --epochs 1000 --backbone mert --variation none --pooling attention --learn_masks
#nohup python3 -u main.py --epochs 1000 --backbone mert --variation none --pooling mean --learn_masks
#nohup python3 -u main.py --epochs 1000 --backbone mert --variation one_layer --pooling attention --learn_masks
#nohup python3 -u main.py --epochs 1000 --backbone mert --variation one_layer --pooling attention 
#nohup python3 -u main.py --epochs 1000 --backbone mert --variation one_layer --pooling mean --learn_masks
#nohup python3 -u main.py --epochs 1000 --backbone mert --variation one_layer --pooling mean
#nohup python3 -u main.py --epochs 1000 --backbone mert --variation mlp --pooling attention --learn_masks
#nohup python3 -u main.py --epochs 1000 --backbone mert --variation mlp --pooling attention 
#nohup python3 -u main.py --epochs 1000 --backbone mert --variation mlp --pooling mean --learn_masks
#nohup python3 -u main.py --epochs 1000 --backbone mert --variation mlp --pooling mean 

#nohup python3 -u main.py --epochs 1000 --backbone muq --variation none --pooling attention --learn_masks
#nohup python3 -u main.py --epochs 1000 --backbone muq --variation none --pooling mean --learn_masks
#nohup python3 -u main.py --epochs 1000 --backbone muq --variation one_layer --pooling attention --learn_masks
#nohup python3 -u main.py --epochs 1000 --backbone muq --variation one_layer --pooling attention 
#nohup python3 -u main.py --epochs 1000 --backbone muq --variation one_layer --pooling mean --learn_masks
#nohup python3 -u main.py --epochs 1000 --backbone muq --variation one_layer --pooling mean
#nohup python3 -u main.py --epochs 1000 --backbone muq --variation mlp --pooling attention --learn_masks
#nohup python3 -u main.py --epochs 1000 --backbone muq --variation mlp --pooling attention 
#nohup python3 -u main.py --epochs 1000 --backbone muq --variation mlp --pooling mean --learn_masks
#nohup python3 -u main.py --epochs 1000 --backbone muq --variation mlp --pooling mean 

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument('--n_conditions', type=int, default=3, help='Number of Conditions')

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch Size')
    parser.add_argument('--epochs', type=int, required=True, help='Number of Training Epochs')

    parser.add_argument('--backbone', type=str, required=True, help="Backbone Model Name: 'mert', 'clap' or 'muq'")
    parser.add_argument('--variation', type=str, required=True, help="Additional Weights for Finetuning Variation: 'one_layer', 'mlp' or 'none'")
    parser.add_argument('--pooling', type=str, required=True, help="Pooling Method: 'mean' or 'attention'")
    parser.add_argument('--learn_masks', action='store_true', help='Whether to learn masks')

    args = parser.parse_args()

    return args