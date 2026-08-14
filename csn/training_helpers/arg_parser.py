import argparse

#nohup python3 -u main.py --backbone clap --lr 1e-5 --batch_size 32 --epochs 100 > clap.out & 

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument('--backbone', type=str, default='clap', help='Backbone Model Name')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=1, help='Number of Training Epochs')

    args = parser.parse_args()

    return args