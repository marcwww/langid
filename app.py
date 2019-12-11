from dataset import NgramFeature, UnicodeBlockFeature
from ffd import FeedforwardNetwork
import os
import torch
import utils
import pandas as pd

mdir = 'mdl/ffd-uniblock/'
ft_extractors = {f'{n}-gram': NgramFeature(n, vsize) for n, vsize in \
                 zip([1, 2, 3, 4], [1000, 1000, 5000, 5000])}
ft_extractors['unicode-block'] = UnicodeBlockFeature()
for name in ft_extractors:
    cache_path = os.path.join(mdir, f'{name}.pkl')
    assert os.path.exists(cache_path)
    ft_extractors[name] = utils.load_obj(cache_path)


def build_batch(txt):
    batch = {}
    for name, extractor in ft_extractors.items():
        if 'gram' in name:
            batch[name] = torch.LongTensor([ft_extractors[name].extract(txt)])
        elif name == 'unicode-block':
            batch[name] = torch.Tensor([ft_extractors[name].extract(txt)])
        else:
            raise NotImplementedError
    return batch


obj_path = os.path.join(mdir, 'lang.pkl')
assert os.path.exists(obj_path)
LANG = utils.load_obj(obj_path)

obj_path = os.path.join(mdir, 'args.pkl')
assert os.path.exists(obj_path)
args = utils.load_obj(obj_path)

mdl = FeedforwardNetwork(args, ft_extractors, LANG)
fmdl = os.path.join(mdir, 'mdl.pkl')
mdl.load_state_dict(torch.load(fmdl, map_location=torch.device('cpu')))
utils.log(f'Loaded model from {fmdl}')

iso_639_4 = pd.read_csv('ISO-639-4.csv', '\t')
lang2label = {row.iso: row.label for idx, row in iso_639_4.iterrows()}
utils.log(f'Loaded ISO-639-4')

while True:
    line = input('Pleasing input text for language identification: ')
    batch = build_batch(line)
    logits = mdl(batch)
    pred = logits.max(dim=-1)[1]
    pred = LANG.decode(pred)[0]
    label = lang2label[pred]
    print(f'Model output: {label} (ISO-649-4: {pred})')

