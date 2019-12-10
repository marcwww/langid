import torch
from torch import nn
import utils
from macros import *


class FeedforwardNetwork(nn.Module):

    def __init__(self, args, ft_extractors: dict, LANG):
        super(FeedforwardNetwork, self).__init__()
        hdim = args.hdim
        ngram_dim = args.ngram_dim
        drop = args.drop
        self.ft_extractors = ft_extractors
        embeddings = {name: utils.DropEmbedding(num_embeddings=len(extractor.itos),
                                                embedding_dim=ngram_dim,
                                                drop=drop) for name, extractor in ft_extractors.items()}
        self.embeddings = nn.ModuleDict(embeddings)
        self.clf = nn.Sequential(nn.Linear(ngram_dim * len(embeddings), hdim),
                                 nn.ReLU(),
                                 nn.Linear(hdim, len(LANG.itos)))
        self.reset_params()
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
            padding_idx = extractor.stoi[PAD]
            mask = batch.eq(padding_idx)
            pooled.append(utils.mean_pooling(emb, mask))
        pooled = torch.cat(pooled, dim=1)  # (bsz, fdim)
        logits = self.clf(pooled)
        return logits

    def calc_loss(self, inp: dict):
        lang = inp['lang']
        logits = self.forward(inp)
        loss = self.criterion(logits, lang)
        return loss