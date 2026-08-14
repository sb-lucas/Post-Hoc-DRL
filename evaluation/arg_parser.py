import argparse

#python3 evaluate.py --backbone clap --path "clap_0617_210313" 
#python3 evaluate.py --backbone clap --path "" 


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str, default='clap')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--path', type=str, default="")

    args = parser.parse_args()

    return args