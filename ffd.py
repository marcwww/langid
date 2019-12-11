import torch
from torch import nn
import utils
from macros import *


class FeedforwardNetwork(nn.Module):

    def __init__(self, args, ft_extractors: dict, LANG):
        super(FeedforwardNetwork, self).__init__()
        hdim = args.hdim
        ngram_dim = args.ngram_dim
        uniblock_dim = args.uniblock_dim
        ngram_drop = args.ngram_drop
        uniblock_drop = args.uniblock_drop
        word_dim = args.word_dim
        word_drop = args.word_drop
        balanced_exp = args.balanced_exp
        self.ft_extractors = ft_extractors
        edims = {}
        drops = {}
        for name in ft_extractors:
            if 'gram' in name:
                edims[name] = ngram_dim
                drops[name] = ngram_drop
            elif name == 'unicode-block':
                edims[name] = uniblock_dim
                drops[name] = uniblock_drop
            elif name == 'word':
                edims[name] = word_dim
                drops[name] = word_drop
            else:
                raise NotImplementedError
        embeddings = {name: utils.DropEmbedding(num_embeddings=len(extractor.itos),
                                                embedding_dim=edims[name],
                                                drop=drops[name]) for name, extractor in ft_extractors.items()}
        self.embeddings = nn.ModuleDict(embeddings)
        self.clf = nn.Sequential(nn.Linear(ngram_dim * sum(['gram' in name for name in ft_extractors]) +
                                           uniblock_dim + word_dim, hdim),
                                 nn.ReLU(),
                                 nn.Linear(hdim, len(LANG.itos)))
        self.reset_params()
        if balanced_exp is not None:
            freq_lang = torch.Tensor(LANG.freq)
            freq_lang = (1 / freq_lang) ** balanced_exp
            self.criterion = nn.CrossEntropyLoss(weight=freq_lang)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def reset_params(self):
        for n, p in list(self.named_parameters()):
            if len(p.shape) > 1:
                p.data.normal_(0, 0.02)
            else:
                p.data.zero_()

    def forward(self, inp: dict):
        pooled = []
        for name, extractor in self.ft_extractors.items():
            batch = inp[name]
            emb = self.embeddings[name](batch)
            if 'gram' in name or name == 'word':
                # batch: (bsz, seq_len, ngrams)
                padding_idx = extractor.stoi[PAD]
                mask = batch.eq(padding_idx)
                pooled.append(utils.mean_pooling(emb, mask))
            elif name == 'unicode-block':
                # batch: (bsz, nblocks)
                pooled.append(emb)
            else:
                raise NotImplementedError

        pooled = torch.cat(pooled, dim=1)  # (bsz, fdim)
        logits = self.clf(pooled)
        return logits

    def calc_loss(self, inp: dict):
        lang = inp['lang']
        logits = self.forward(inp)
        loss = self.criterion(logits, lang)
        return loss