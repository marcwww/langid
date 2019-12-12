from dataset import NgramFeature, UnicodeBlockFeature
from ffd import FeedforwardNetwork
import os
import torch
import utils
import pandas as pd
import argparse
import tqdm
from sklearn.metrics import f1_score, precision_score, \
    recall_score, accuracy_score, classification_report

mdir = 'mdl/ffd-word-balanced'
ft_names = ['1-gram', '2-gram', '3-gram', '4-gram', 'unicode-block', 'word']
ft_extractors = {name: utils.load_obj(os.path.join('cache', name + '.pkl')) for name in ft_names}


def build_batch(txt):
    batch = {}
    for name, extractor in ft_extractors.items():
        if 'gram' in name:
            batch[name] = torch.LongTensor([ft_extractors[name].extract(txt)])
        elif name == 'unicode-block':
            batch[name] = torch.Tensor([ft_extractors[name].extract(txt)])
        elif name == 'word':
            batch[name] = torch.LongTensor([ft_extractors[name].extract(txt)])
        else:
            raise NotImplementedError
    return batch


obj_path = os.path.join('cache', 'lang.pkl')
assert os.path.exists(obj_path)
LANG = utils.load_obj(obj_path)

obj_path = os.path.join(mdir, 'args.pkl')
assert os.path.exists(obj_path)
args = utils.load_obj(obj_path)

mdl = FeedforwardNetwork(args, ft_extractors, LANG)
fmdl = os.path.join(mdir, 'mdl.pkl')
mdl.load_state_dict(torch.load(fmdl, map_location=torch.device('cpu')))
mdl.eval()
utils.log(f'Loaded model from {fmdl}')

iso_639_4 = pd.read_csv('ISO-639-4.csv', '\t')
lang2label = {row.iso: row.label for idx, row in iso_639_4.iterrows()}
utils.log(f'Loaded ISO-639-4')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    # ------------ high-level argument ------------
    parser.add_argument("-ftest", required=True, type=str, help="In a CSV format with header of 'lang'\t'txt'")
    args = parser.parse_args()
    df = pd.read_csv(args.ftest, '\t')
    preds = []
    golds = []
    with torch.no_grad():
        for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            batch = build_batch(row.txt)
            logits = mdl(batch)
            pred = logits.max(dim=-1)[1]
            pred = LANG.decode(pred)[0]
            preds.append(pred)
            golds.append(row.lang)

    acc = accuracy_score(golds, preds) * 100
    f1 = f1_score(golds, preds, average='macro') * 100
    precision = precision_score(golds, preds, average='macro') * 100
    recall = recall_score(golds, preds, average='macro') * 100
    res = {'acc': round(acc, 2),
           'f1': round(f1, 2),
           'precision': round(precision, 2),
           'recall': round(recall, 2)}

    utils.log(f'Test result for {args.ftest}: {res}')

