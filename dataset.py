import re
import pandas as pd
import torch
from collections import Counter
from macros import *
import utils
import os
import tqdm
from collections import defaultdict
import numpy as np


class WordFeature(object):

    def __init__(self, vsize):
        self.itos = []
        self.stoi = {}
        self.vsize = vsize

    def tokenize(self, txt):
        return txt.lower().split()

    def build(self, txts):
        counter = defaultdict(int)
        for txt in tqdm.tqdm(txts):
            for word in self.tokenize(txt):
                counter[word] += 1
        counter = sorted(counter.items(),
                         key=lambda x: x[1],
                         reverse=True)[:self.vsize]
        self.add(PAD)
        for ngram, c in counter:
            self.add(ngram)

    def add(self, ngram):
        if ngram in self.stoi:
            return
        self.itos.append(ngram)
        self.stoi[ngram] = len(self.itos) - 1

    def extract(self, txt):
        return [self.stoi[word] for word in self.tokenize(txt) if word in self.stoi]


class NgramFeature(object):

    def __init__(self, n, vsize):
        self.n = n
        self.vsize = vsize
        self.itos = []
        self.stoi = {}

    def tokenize(self, txt):
        ngrams = utils.extract_ngrams(txt, self.n)
        return filter(lambda ngram: len(''.join(ngram.split())) == self.n, ngrams)

    def build(self, txts):
        counter = defaultdict(int)
        for txt in tqdm.tqdm(txts):
            for ngram in self.tokenize(txt):
                counter[ngram] += 1
        counter = sorted(counter.items(),
                         key=lambda x: x[1],
                         reverse=True)[:self.vsize]
        self.add(PAD)
        for ngram, c in counter:
            self.add(ngram)

    def add(self, ngram):
        if ngram in self.stoi:
            return
        self.itos.append(ngram)
        self.stoi[ngram] = len(self.itos) - 1

    def extract(self, txt):
        ngrams = self.tokenize(txt)
        return [self.stoi[ngram] for ngram in ngrams if ngram in self.stoi]


class UnicodeBlockFeature(object):

    def __init__(self):
        self.bs = None
        self.es = None
        self.itos = []
        self.stoi = {}

    def build(self, ftable):
        # table: begin end label
        table = pd.read_csv(ftable, '\t')
        self.bs = np.array([[int(n, 16) for n in table.begin]])
        self.es = np.array([[int(n, 16) for n in table.end]])
        for uniblock in table.label:
            self.add(uniblock)

    def add(self, uniblock):
        if uniblock in self.stoi:
            return
        self.itos.append(uniblock)
        self.stoi[uniblock] = len(self.itos) - 1

    def extract(self, txt):
        # return normalized frequency vector
        txt = ''.join(txt.split())
        unicodes = np.array([[ord(ch)] for ch in txt])
        locate = (unicodes >= self.bs) & (unicodes < self.es)  # (seq_len, nblocks)
        vec = locate.sum(axis=0)
        vec_sum = vec.sum()
        if vec_sum > 0:
            return vec / vec_sum
        return vec


class Lang(object):

    def __init__(self):
        self.stoi = {}
        self.itos = []
        self.freq = None

    def build(self, langs):
        freqs = Counter(langs)
        freqs = dict(sorted(freqs.items(), key=lambda x: x[1]))
        for lang in freqs:
            self.add(lang)
        self.freq = [freqs[lang] for lang in self.itos]

    def add(self, lang):
        if lang in self.stoi:
            return
        self.itos.append(lang)
        self.stoi[lang] = len(self.itos) - 1

    def encode(self, langs):
        return list(map(lambda s: self.stoi[s], langs))

    def decode(self, indices):
        return list(map(lambda i: self.itos[i], indices))


class LangIDDataset(object):

    def __init__(self, args):
        device = torch.device(args.gpu if args.gpu != -1 else 'cpu')
        self.device = device
        ftrain = args.ftrain
        fvalid = args.fvalid
        ftest = args.ftest
        futable = args.futable
        bsz = args.bsz
        train, valid, test = self.load_data(ftrain), \
                             self.load_data(fvalid), \
                             self.load_data(ftest)
        ft_extractors = {f'{n}-gram': NgramFeature(n, vsize) for n, vsize in \
                         zip([1, 2, 3, 4], [10000, 10000, 50000, 50000])}
        ft_extractors['unicode-block'] = UnicodeBlockFeature()
        ft_extractors['word'] = WordFeature(50000)
        for name in ft_extractors:
            cache_path = os.path.join('cache', f'{name}.pkl')
            if os.path.exists(cache_path):
                ft_extractors[name] = utils.load_obj(cache_path)
            else:
                utils.log(f'Building feature {name}')
                if 'gram' in name:
                    ft_extractors[name].build(train.txt)
                elif name == 'unicode-block':
                    ft_extractors[name].build(futable)
                elif name == 'word':
                    ft_extractors[name].build(train.txt)
                else:
                    raise NotImplementedError
                utils.save_obj(ft_extractors[name], cache_path)

        cache_path = os.path.join('cache', 'lang.pkl')
        LANG = Lang()
        if os.path.exists(cache_path):
            LANG = utils.load_obj(cache_path)
        else:
            utils.log('Building LANG')
            LANG.build(train.lang)
            utils.save_obj(LANG, cache_path)

        utils.log('Building batches')
        self.train_iter = self.build_batches(train, 'train', ft_extractors, bsz, LANG, True, device)
        self.valid_iter = self.build_batches(valid, 'valid', ft_extractors, bsz, LANG, False, device)
        self.test_iter = self.build_batches(test, 'test', ft_extractors, bsz, LANG, False, device)
        self.ft_extractors = ft_extractors
        self.LANG = LANG

    @staticmethod
    def load_data(fpath):
        utils.log(f'Loading data from {fpath}')
        df = pd.read_csv(fpath, '\t')
        df.set_index('id', inplace=True)
        df['len'] = [len(txt) for txt in df.txt]
        return df

    @staticmethod
    def build_batch_ngram(txts, ft_extractor: NgramFeature):
        seqs = [ft_extractor.extract(txt) for txt in txts]
        pad_idx = ft_extractor.stoi[PAD]
        seq_len = max([len(seq) for seq in seqs])
        for seq in seqs:
            seq.extend([pad_idx] * (seq_len - len(seq)))
        return seqs

    @staticmethod
    def build_batch_uniblock(txts, ft_extractor: UnicodeBlockFeature):
        return [ft_extractor.extract(txt) for txt in txts]

    @staticmethod
    def build_batch_word(txts, ft_extractor: WordFeature):
        seqs = [ft_extractor.extract(txt) for txt in txts]
        pad_idx = ft_extractor.stoi[PAD]
        seq_len = max([len(seq) for seq in seqs])
        for seq in seqs:
            seq.extend([pad_idx] * (seq_len - len(seq)))
        return seqs

    @staticmethod
    def build_batches(df, df_type, ft_extractors: dict, bsz, LANG: Lang, shuffle, device):
        data = {name: [] for name in ft_extractors}
        data['lang'] = []

        df = df.sort_values(by='len')
        df.lang = LANG.encode(df.lang)
        indices = np.arange(len(df))
        bs = indices[::bsz]
        es = bs + bsz
        es[es > len(indices)] = len(indices)
        index_segs = [indices[b:e] for b, e in zip(bs, es)]
        if shuffle:
            np.random.shuffle(index_segs)

        for name in data:
            cache_path = os.path.join('cache', f'{df_type}.{name}.cache')

            if os.path.exists(cache_path):
                data[name] = utils.load_obj(cache_path)
            else:
                utils.log(f'Extracting {name}')
                for seg in tqdm.tqdm(index_segs):
                    df_seg = df.iloc[seg]
                    if name == 'lang':
                        batch = df_seg.lang.tolist()
                    elif 'gram' in name:
                        batch = LangIDDataset.build_batch_ngram(df_seg.txt, ft_extractors[name])
                    elif name == 'unicode-block':
                        batch = LangIDDataset.build_batch_uniblock(df_seg.txt, ft_extractors[name])
                    elif name == 'word':
                        batch = LangIDDataset.build_batch_word(df_seg.txt, ft_extractors[name])
                    else:
                        raise NotImplementedError
                    data[name].append(batch)
                utils.save_obj(data[name], cache_path)

        tensor_types = {}
        for name in data:
            if name == 'unicode-block':
                tensor_types[name] = torch.Tensor
            else:
                tensor_types[name] = torch.LongTensor

        return utils.BatchIterator(data, len(index_segs), tensor_types, device)
