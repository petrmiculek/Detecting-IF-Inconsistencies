import os
import random
import re
from os.path import join
from math import ceil
from collections.abc import Iterable

# external
from gensim.models import FastText
import torch
import numpy as np
import pandas as pd
import libcst as cst

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, Subset, random_split, DataLoader

# local
from src import config
from src.preprocess import rebuild_cond, negate_cond, load_tokenizer
from src.util import real_len

DO_NOTHING = 0
FLIP_CONDITION = 1


class IfRaisesDataset(Dataset):
    def __init__(self, path, tokenizer, transform=None, fraction=1.0):
        assert os.path.isfile(path), \
            'Dataset could not find path "{}"'.format(path)

        self.dataset = pd.read_pickle(path)
        self.transform = transform
        self.tokenizer = tokenizer
        self.fraction = fraction

        if fraction != 1.0:
            end = int(ceil(len(self.dataset) * fraction))
            print('Using {} out of {} dataset samples'.format(end, len(self.dataset)))
            self.dataset = self.dataset[:end]

        self.ds_len = len(self.dataset)

        self.sep_token_id = self.tokenizer.token_to_id("</s>")
        self.sep_token_str = self.tokenizer.id_to_token(self.sep_token_id)

        self.embed_model = FastText.load(config.fasttext_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return (self[i] for i in range(start, stop, step))

        elif isinstance(idx, Iterable):
            return (self.__getitem__(i) for i in idx)
        elif not np.issubdtype(type(idx), np.integer):
            return TypeError('Invalid index type')

        sample = self.dataset.iloc[idx]  # recombination here

        if self.transform:
            sample = self.transform(sample)

        cond_node = sample['cond']
        is_else = sample['else']
        raise_node = sample['raise']
        context_str = sample['context']

        twist = np.random.randint(0, 2)

        if twist == DO_NOTHING:
            # no twist
            pass
        elif twist == FLIP_CONDITION:
            # negate
            is_else = not is_else
        else:
            diff_idx = np.random.randint(0, self.ds_len)
            diff_sample = self.dataset.iloc[diff_idx]
            diff_cond_node = diff_sample['cond']

            # check if cond is the same
            cond_node = diff_cond_node

        # fix cond
        if is_else:
            cond_node = negate_cond(cond_node)
        else:
            cond_node = rebuild_cond(cond_node, parentheses=False)

        # get str repr
        cond_str = cst.Module([cond_node]).code
        raise_str = cst.Module([raise_node]).code

        full_str = " ".join([cond_str, self.sep_token_str, raise_str])

        tokens_cond = self.tokenizer.encode(cond_str)
        tokens_raise = self.tokenizer.encode(raise_str)
        # tokens_context = tokenizer.encode(context_str)

        tokens_all = self.tokenizer.encode(full_str)

        # lengths.append({
        #     'cond': real_len(tokens_cond),
        #     'raise': real_len(tokens_raise),
        #     'all': real_len(tokens_all)
        # })

        tokens_both = tokens_cond.tokens + tokens_raise.tokens[1:]  # skip start token for 'sentence 2'
        embeds = self.embed_model.wv[tokens_both]
        # embeddings.append(token_embeds)
        # embeds_raise = embed_model.wv[tokens_raise.tokens]

        if twist == DO_NOTHING:
            target = config.consistent
        else:
            target = config.inconsistent

        return embeds, target
