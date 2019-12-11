import pickle
import logging
import torch
import random
import numpy as np
from torch import nn
from torch.nn import init
import time
import pandas as pd
from collections import defaultdict

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


def log(msg):
    logging.info(msg)


def save_txt(txt, fpath):
    with open(fpath, 'w') as f:
        f.write(txt)
    log(f'Saved to {fpath}')


def save_obj(obj, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f)
    log(f'Saved to {fpath}')


def load_obj(fpath):
    with open(fpath, 'rb') as f:
        obj = pickle.load(f)
    log(f'Loaded from {fpath}')
    return obj


def init_seed(seed=None):
    """Seed the RNGs for predicatability/reproduction purposes."""

    if seed is None:
        # seed = int(get_ms() // 1000)
        seed = eval(''.join(list(str(int(time.time() * 1e6)))[-6:]))

    log(f"Using seed={seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True


def extract_ngrams(toks, n):
    return map(lambda x: ''.join(x), zip(*[toks[i:] for i in range(n)]))


class DropEmbedding(nn.Embedding):

    def __init__(self, **kwargs):
        drop = kwargs.pop('drop')
        super(DropEmbedding, self).__init__(**kwargs)
        self.drop = nn.Dropout(drop)
        self.reset_params()

    def reset_params(self):
        init.normal_(self.weight, std=0.01)

    def forward(self, input):
        if input.dtype == torch.int64:
            return self.drop(super(DropEmbedding, self).forward(input))
        else:
            assert input.dtype == torch.float32
            return self.drop(input.matmul(self.weight))


def mean_pooling(hids: torch.Tensor, mask_eq_pad):
    # hids: (bsz, seq_len, hdim)
    # mask_eq_pad: (bsz, seq_len)
    lengths = mask_eq_pad.sum(dim=-1)
    lengths = lengths.masked_fill(lengths.eq(0), 1).float().unsqueeze(-1)
    pooled = hids.masked_fill(mask_eq_pad.unsqueeze(-1), 0).sum(dim=1) / lengths
    return pooled  # (bsz, hdim)


class BatchIterator(object):

    def __init__(self, df_raw: pd.DataFrame, segs, build_batch):
        self.df_raw = df_raw
        self.segs = segs
        self.build_batch = build_batch
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def __len__(self):
        return len(self.segs)

    def __next__(self):
        while self.i < len(self.segs):
            seg = self.segs[self.i]
            df_seg = self.df_raw.iloc[seg]
            batch = self.build_batch(df_seg)
            self.i += 1
            return batch
        raise StopIteration


class DataIterator(object):
    def __init__(self, data: dict, device):
        self.data = data
        lens = set([len(v) for k, v in data.items() if k != 'uinp'])
        assert len(lens) == 1
        self.num = list(lens)[0]
        self.i = 0
        self.i_global = 0
        self.device = device

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < self.num:
            # Note here mod can make the values in different tracking ways,
            # especially for the unlabeled data.
            d = {k: v[self.i_global % len(v)].to(self.device) for k, v in self.data.items()}
            self.i += 1
            self.i_global += 1
            return d
        else:
            raise StopIteration

    def __len__(self):
        return self.num


class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = defaultdict()

    def _insert(self, word, val):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        current = self.root
        for letter in word:
            current = current.setdefault(letter, {})
        current.setdefault("_end", val)

    def _search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        current = self.root
        for letter in word:
            if letter not in current:
                return None
            current = current[letter]
        if "_end" in current:
            return current['_end']
        return None

    def __getitem__(self, key):
        val = self._search(key)
        if val is None:
            raise KeyError
        return val

    def __setitem__(self, key, val):
        self._insert(key, val)

    def from_dict(self, d):
        for key, val in d.items():
            self._insert(key, val)

    def start_with(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        current = self.root
        for letter in prefix:
            if letter not in current:
                return False
            current = current[letter]
        return True
