import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import torch
from collections import Counter
from macros import *
import utils
import os
import tqdm
import argparse
from dataset import LangIDDataset
from ffd import FeedforwardNetwork
from torch import optim
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import f1_score, precision_score, \
    recall_score, accuracy_score, classification_report
import numpy as np
import crash_on_ipy


def run_iter(mdl: FeedforwardNetwork, batch, optimizer, is_training, args):
    mdl.train(is_training)
    if is_training:
        loss = mdl.calc_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        grad = clip_grad_norm_(mdl.parameters(), args.gclip)
        optimizer.step()
        return loss.item()
    else:
        logits = mdl(batch)
        pred = logits.max(dim=-1)[1]
        return pred


def valid(mdl: FeedforwardNetwork, data_iter: utils.BatchIterator, LANG, args):
    with torch.no_grad():
        preds = []
        golds = []
        for i, batch in enumerate(tqdm.tqdm(data_iter, total=data_iter.__len__())):
            pred = run_iter(mdl, batch, None, False, None)
            preds.extend(LANG.decode(pred))
            golds.extend(LANG.decode(batch['lang']))

    acc = accuracy_score(golds, preds) * 100
    f1 = f1_score(golds, preds, average='macro') * 100
    precision = precision_score(golds, preds, average='macro') * 100
    recall = recall_score(golds, preds, average='macro') * 100
    res = {'acc': round(acc, 2),
           'f1': round(f1, 2),
           'precision': round(precision, 2),
           'recall': round(recall, 2)}
    report = classification_report(golds, preds, digits=4)
    utils.save_txt(report, os.path.join(args.mdir,
                                        f'report-acc{acc:.2}-'
                                        f'f1{f1:.2}-'
                                        f'p{precision:.2}-'
                                        f'r{recall:.2}.txt'))
    return res


def train(args):
    dataset = LangIDDataset(args)
    mdl = FeedforwardNetwork(args, dataset.ft_extractors, dataset.LANG).to(dataset.device)
    optimizer = optim.Adam(params=[p for p in mdl.parameters() if p.requires_grad],
                           lr=args.lr, weight_decay=args.l2reg)
    # optimizer = optim.ASGD(params=[p for p in mdl.parameters() if p.requires_grad],
    #                        lr=1e-1, weight_decay=args.l2reg)

    utils.log('Begin training')
    valid_res = valid(mdl, dataset.valid_iter, dataset.LANG, args)
    utils.log(f'Initial ' + str(valid_res))
    best_perf = valid_res['f1']
    log_every = len(dataset.train_iter) // 10
    for epoch in range(args.nepoches):
        train_iter = tqdm.tqdm(dataset.train_iter)
        losses = []
        for i, batch in enumerate(train_iter):
            try:
                loss = run_iter(mdl, batch, optimizer, True, args)
                losses.append(loss)
                train_iter.set_description(f'Epoch {epoch} '
                                           f'Loss {loss:.4f}')
            except:
                utils.log('Ignored one iteration due to exception')

            if (i + 1) % log_every == 0:
                valid_res = valid(mdl, dataset.valid_iter, dataset.LANG, args)
                utils.log(f'Epoch {epoch} '
                          f'Loss {np.mean(losses):.4f} ' + str(valid_res))
                losses = []
                if valid_res['f1'] > best_perf:
                    best_perf = valid_res['f1']
                    fmdl = os.path.join(args.mdir, 'mdl.pkl')
                    torch.save(mdl.state_dict(), fmdl)
                    utils.log(f'Saved model to {fmdl}')
    utils.log('Training done')
    test_res = valid(mdl, dataset.test_iter, dataset.LANG, args)
    utils.log(f'Test result: {test_res}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("-nepoches", default=4, type=int)
    parser.add_argument("-ngram_dim", default=16, type=int)
    parser.add_argument("-uniblock_dim", default=8, type=int)
    parser.add_argument("-word_dim", default=16, type=int)
    parser.add_argument("-hdim", default=256, type=int)
    parser.add_argument("-bsz", default=256, type=int)
    parser.add_argument("-ngram_drop", default=0.01, type=float)
    parser.add_argument("-uniblock_drop", default=0.01, type=float)
    parser.add_argument("-word_drop", default=0.5, type=float)
    parser.add_argument("-gclip", default=1, type=float)
    parser.add_argument('-balanced_exp', default=0.1, type=float)
    parser.add_argument("-lr", default=1e-3, type=float)
    parser.add_argument("-l2reg", default=0, type=float)
    parser.add_argument("-gpu", default=-1, type=int)
    parser.add_argument('-seed', default=42, type=int)

    parser.add_argument("-mdir", type=str, default='mdl/ffd')
    parser.add_argument('-ftrain', default='data/train.csv', type=str)
    parser.add_argument('-fvalid', default='data/valid.csv', type=str)
    parser.add_argument('-ftest', default='data/test.csv', type=str)
    parser.add_argument('-futable', default='data/unicode_blocks.csv', type=str)

    args = parser.parse_args()
    utils.init_seed(args.seed)
    utils.log(f'Params: {str(args)}')

    if os.path.exists(args.mdir):
        utils.log(f'{args.mdir} already exists. '
                  f'Input \'yes\' to overwrite, '
                  f'or \'no\' to load and train:')
        key = input()
        if key == 'yes':
            os.system(f'rm -rf {args.mdir}')
            os.makedirs(args.mdir)
        elif key == 'no':
            pass
        else:
            exit()
    else:
        os.makedirs(args.mdir)

    assert os.path.exists(args.mdir)
    utils.save_obj(args, os.path.join(args.mdir, 'args.pkl'))
    train(args)
