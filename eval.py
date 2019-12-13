from dataset import *
from ffd import FeedforwardNetwork
import os
import torch
import utils
import pandas as pd
import argparse
import tqdm
from sklearn.metrics import f1_score, precision_score, \
    recall_score, accuracy_score, classification_report
import crash_on_ipy
import warnings
warnings.filterwarnings("ignore")


def build_batch(txt, ft_extractors):
    batch = {}
    for name, extractor in ft_extractors.items():
        if name == 'unicode-block':
            tensor_type = torch.Tensor
        else:
            tensor_type = torch.LongTensor
        batch[name] = tensor_type([extractor.extract(txt)])
    return batch


def load_whole(mdir):
    # mdir = 'mdl/ffd-googledrop-256hdim'
    ft_names = ['1-gram', '2-gram', '3-gram', '4-gram', 'unicode-block', 'word']
    ft_extractors = {name: None for name in ft_names}
    # ft_extractors = {f'{n}-gram': NgramFeature(n, vsize) for n, vsize in \
    #                  zip([1, 2, 3, 4], [10000, 10000, 50000, 50000])}
    # ft_extractors['unicode-block'] = UnicodeBlockFeature()
    # ft_extractors['word'] = WordFeature(50000)
    for name in ft_extractors:
        cache_path = os.path.join('cache', f'{name}.pkl')
        assert os.path.exists(cache_path)
        ft_extractors[name] = utils.load_obj(cache_path)

    cache_path = os.path.join('cache', 'lang.pkl')
    assert os.path.exists(cache_path)
    LANG = utils.load_obj(cache_path)

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

    return ft_extractors, LANG, args, mdl, lang2label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("-ftest", required=True, type=str, help="In a CSV format with header of 'lang'\t'txt'")
    parser.add_argument("-mdir", default='mdl/ffd-googledrop-256hdim', type=str, help='Indicating the model directory.')
    parser.add_argument("-gpu", default=-1, type=int, help='Indicating GPU index, -1 for CPU')
    parser.add_argument("-bsz", default=256, type=int, help='Indicating batch size')

    args = parser.parse_args()
    device = torch.device(args.gpu if args.gpu != -1 else 'cpu')
    ft_extractors, LANG, _, mdl, lang2label = load_whole(args.mdir)
    mdl.to(device)

    df = pd.read_csv(args.ftest, '\t')
    preds = []
    golds = []
    labels = []
    with torch.no_grad():
        data = LangIDDataset.load_data(args.ftest)
        data_iter, _ = LangIDDataset.build_batches(data, 'test', ft_extractors, args.bsz, LANG, False, device)
        for batch in tqdm.tqdm(data_iter):
            # mdl.train(False)
            logits = mdl(batch)
            pred = logits.max(dim=-1)[1]
            pred = LANG.decode(pred)
            preds.extend(pred)
            golds.extend(LANG.decode(batch['lang']))
            label = [lang2label[p] for p in pred]
            labels.extend(label)
    for suffix in ['.1-gram.cache', '.2-gram.cache', '.3-gram.cache', '.4-gram.cache',
                   '.lang.cache', '.unicode-block.cache', '.word.cache']:
        cache_path = os.path.join('cache', 'test' + suffix)
        os.system(f'rm {cache_path}')
        utils.log(f'Cleared cache {cache_path}')

    acc = accuracy_score(golds, preds) * 100
    f1 = f1_score(golds, preds, average='macro') * 100
    precision = precision_score(golds, preds, average='macro') * 100
    recall = recall_score(golds, preds, average='macro') * 100
    res = {'acc': round(acc, 2),
           'f1': round(f1, 2),
           'precision': round(precision, 2),
           'recall': round(recall, 2)}
    utils.log(f'Test result for {args.ftest}: {res}')