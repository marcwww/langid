from dataset import NgramFeature, UnicodeBlockFeature
from ffd import FeedforwardNetwork
import os
import torch
import utils
import pandas as pd
import argparse
import tqdm
from dataset import *
from eval import load_whole


def load_data(fpath):
    utils.log(f'Loading data from {fpath}')
    df = pd.read_csv(fpath, '\t', names=['txt'])
    # df.set_index('id', inplace=True)
    df['len'] = [len(txt) for txt in df.txt]
    df['lang'] = ['cmn'] * len(df)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("-finput", required=True, type=str, help="One line one input")
    parser.add_argument("-mdir", default='mdl/ffd-googledrop-256hdim', type=str, help='Indicating the model directory.')
    parser.add_argument("-gpu", default=-1, type=int, help='Indicating GPU index, -1 for CPU')
    parser.add_argument("-bsz", default=256, type=int, help='Indicating batch size')
    args = parser.parse_args()

    device = torch.device(args.gpu if args.gpu != -1 else 'cpu')
    ft_extractors, LANG, _, mdl, lang2label = load_whole(args.mdir)
    mdl.to(device)

    preds = []
    labels = []
    data_type = 'input-only'
    with torch.no_grad():
        data = load_data(args.finput)
        data_iter, df = LangIDDataset.build_batches(data, data_type, ft_extractors, args.bsz, LANG, False, device)
        for batch in tqdm.tqdm(data_iter):
            mdl.train(False)
            logits = mdl(batch)
            pred = logits.max(dim=-1)[1]
            pred = LANG.decode(pred)
            preds.extend(pred)
            label = [lang2label[p] for p in pred]
            labels.extend(label)
    for suffix in ['.1-gram.cache', '.2-gram.cache', '.3-gram.cache', '.4-gram.cache',
                   '.lang.cache', '.unicode-block.cache', '.word.cache']:
        cache_path = os.path.join('cache', data_type + suffix)
        os.system(f'rm {cache_path}')
        utils.log(f'Cleared cache {cache_path}')

    fout = 'out.txt'
    df = pd.DataFrame({'lang': labels, 'ISO-639-4': preds, 'input': df.txt, 'id': df.index})
    df.sort_values(by='id', inplace=True)
    df.set_index('id', inplace=True)
    df.to_csv(fout, '\t')
    utils.log(f'Predictions are saved to {fout}')
