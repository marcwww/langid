from dataset import NgramFeature, UnicodeBlockFeature
from ffd import FeedforwardNetwork
import os
import torch
import utils
import pandas as pd
import argparse
from dataset import *
import eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("-mdir", default='mdl/ffd-googledrop-256hdim', type=str, help='Indicating the model directory.')
    parser.add_argument("-gpu", default=-1, type=int, help='Indicating GPU index, -1 for CPU')
    parser.add_argument("-bsz", default=256, type=int, help='Indicating batch size')

    args = parser.parse_args()
    device = torch.device(args.gpu if args.gpu != -1 else 'cpu')
    ft_extractors, LANG, _, mdl, lang2label = eval.load_whole(args.mdir)
    mdl.to(device)

    while True:
        with torch.no_grad():
            line = input('Pleasing input text for language identification: ')
            if len(line.strip()) == 0:
                continue
            batch = eval.build_batch(line, ft_extractors)
            batch = {name: val.to(device) for name, val in batch.items()}
            logits = mdl(batch)
            pred = logits.max(dim=-1)[1]
            pred = LANG.decode(pred)[0]
            label = lang2label[pred]
            print(f'Model output: {label} (ISO-649-4: {pred})')


