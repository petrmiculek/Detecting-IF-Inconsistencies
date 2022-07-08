import os
import random
import re
from os.path import join
from math import ceil
from collections.abc import Iterable
from contextlib import suppress

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
from src.extract import load_segments, extract_raises
from src.preprocess import rebuild_cond, negate_cond, load_tokenizer
from src.util import real_len, r

# twist_enable - whether to twist
DO_NOTHING = 0
DO_TWIST = 1
# twist - which twist to use
NO_TWIST = 0  # only indicates no-twist label for the resulting sample - not an option for picking a twist
FLIP_CONDITION = 1
RECOMBINE_COND = 2
RECOMBINE_RAISE = 3
from src.util import repr_cond_raise


class IfRaisesDataset(Dataset):
    def __init__(self, path: str, tokenizer, transform=None, fraction=1.0, eval_mode=False):
        assert os.path.isfile(path), \
            'Dataset could not find path "{}"'.format(path)

        if path.endswith('.pkl'):
            # dataframe with already extracted if-raises
            self.dataset = pd.read_pickle(path)
        elif path.endswith('.json'):
            # prediction scenario
            segments = load_segments(archive=None, file=path)
            samples = extract_raises(segments, max=None)

            self.dataset = pd.DataFrame(samples)
        else:
            raise ValueError('Unknown dataset format: {}'.format(path))  # if-raise, how meta

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

        # debug
        self.human_eval = False
        self.human_preds = []
        self.human_gt = []

        self.eval_mode = eval_mode
        self.triplet_mode = False

    def __len__(self):
        return len(self.dataset)

    def getitem_eval(self, idx):
        sample = self.dataset.iloc[idx]  # recombination here
        cond_node = sample['cond']
        is_else = sample['else']
        raise_node = sample['raise']
        context_str = sample['context']
        line_raise = sample['line_raise']

        # fix cond
        if is_else:
            cond_node = negate_cond(cond_node)
        else:
            cond_node = rebuild_cond(cond_node, parentheses=False)

        cond_str = r(cond_node)
        raise_str = r(raise_node)

        tokens_cond = self.tokenizer.encode(cond_str)
        tokens_raise = self.tokenizer.encode(raise_str)

        tokens_both = tokens_cond.tokens + tokens_raise.tokens[1:]  # skip start token for 'sentence 2'
        embeds = self.embed_model.wv[tokens_both]

        target = config.consistent  # watch out when loading

        return embeds, target, line_raise

    def getitem_triplet(self, idx):
        idx2, idx3 = np.random.randint(0, self.ds_len - 1, size=2)
        # todo get a anchor, positive, negative triplet
        return None, None, None

    def __getitem__(self, idx):
        """
        Returns:

        """
        if self.eval_mode:
            return self.getitem_eval(idx)
        # ====================================================
        if self.triplet_mode:
            return self.getitem_triplet(idx)
        # ====================================================

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return (self[i] for i in range(start, stop, step))

        elif isinstance(idx, Iterable):
            return (self.__getitem__(i) for i in idx)
        elif not np.issubdtype(type(idx), np.integer):
            return TypeError('Invalid index type')

        sample = self.dataset.iloc[idx]  # recombination here

        cond_node = sample['cond']
        is_else = sample['else']
        raise_node = sample['raise']
        context_str = sample['context']

        # fix cond
        if is_else:
            cond_node = negate_cond(cond_node)
        else:
            cond_node = rebuild_cond(cond_node, parentheses=False)

        # debug
        if self.human_eval:
            cond_node_orig = cond_node
            raise_node_orig = raise_node
            cond_str_orig = r(cond_node_orig)
            raise_str_orig = r(raise_node_orig)

        cond_node, raise_node, twist = self.twist_sample(idx, cond_node, raise_node)

        # get str repr

        cond_str = r(cond_node)
        raise_str = r(raise_node)

        # full_str = " ".join([cond_str, self.sep_token_str, raise_str])

        # debug
        if self.human_eval:
            orig = repr_cond_raise(cond_node_orig, raise_node_orig)
            res = repr_cond_raise(cond_node, raise_node)
            print(context_str)
            print('v' * 7)
            print(orig)
            # human_pred = int(input())
            # self.human_preds.append(human_pred)
            self.human_gt.append(twist)
            print(res)
            print(f"<<{twist}>>")
            print('=' * 10)
            _ = input()

        tokens_cond = self.tokenizer.encode(cond_str)
        tokens_raise = self.tokenizer.encode(raise_str)
        # tokens_context = tokenizer.encode(context_str)

        # tokens_all = self.tokenizer.encode(full_str)

        # lengths.append({
        #     'cond': real_len(tokens_cond),
        #     'raise': real_len(tokens_raise),
        #     'all': real_len(tokens_all)
        # })

        tokens_both = tokens_cond.tokens + tokens_raise.tokens[1:]  # skip start token for 'sentence 2'
        embeds = self.embed_model.wv[tokens_both]
        # embeddings.append(token_embeds)
        # embeds_raise = embed_model.wv[tokens_raise.tokens]

        if twist == NO_TWIST:
            target = config.consistent
        else:
            target = config.inconsistent

        with suppress(AttributeError):
            if raise_node.exc.func.value == 'NotImplementedError':
                # NotImplementedError cannot be inconsistent, overwrite label
                target = config.consistent
        with suppress(AttributeError):
            if raise_node.exc.value.value == 'NotImplementedError':
                target = config.consistent

        return embeds, target

    def twist_sample(self, idx, cond_node, raise_node, force_twist=None):
        """
        Create inconsistent data samples.

        :param idx: data sample index
        :param cond_node: condition node
        :param raise_node: raise statement node
        :param force_twist: force a twist (or no twist), overriding the random choice
        """
        if force_twist is None:
            twist_enable = np.random.choice([DO_NOTHING, DO_TWIST])
            twist = np.random.choice([FLIP_CONDITION, RECOMBINE_COND, RECOMBINE_RAISE],
                                     p=[0.5, 0.25, 0.25])
        else:
            twist_enable = DO_NOTHING if force_twist == DO_NOTHING else DO_TWIST
            twist = force_twist

        # twists - part 1/2
        if twist_enable == DO_TWIST:
            if twist == FLIP_CONDITION:
                # negate
                cond_node = negate_cond(cond_node)

            if twist == RECOMBINE_RAISE:
                # swap in another raise
                same_raise = True

                while same_raise:
                    diff_idx = np.random.randint(0, self.ds_len)
                    diff_sample = self.dataset.iloc[diff_idx]
                    diff_raise_node = diff_sample['raise']

                    # check if raise is the same
                    diff_raise_str = r(diff_raise_node)
                    if diff_idx == idx or diff_raise_str != r(raise_node):
                        same_raise = False
                        raise_node = diff_raise_node
                    else:
                        pass
                        # print(f'same raise: {idx}x{diff_idx}',
                        #       diff_raise_str)

            elif twist == RECOMBINE_COND:
                # swap in another cond
                same_cond = True

                while same_cond:
                    diff_idx = np.random.randint(0, self.ds_len)
                    diff_sample = self.dataset.iloc[diff_idx]
                    diff_cond_node = diff_sample['cond']

                    # check if cond is the same
                    diff_cond_node = rebuild_cond(diff_cond_node, parentheses=False)
                    diff_cond_str = r(diff_cond_node)
                    if diff_idx == idx or diff_cond_str != r(cond_node):
                        same_cond = False
                        cond_node = diff_cond_node
                    else:
                        pass
                        # print(f'same cond: {idx}x{diff_idx}',
                        #       diff_cond_str)

        # careful about indentation
        else:
            # twist_enable == DO_NOTHING => twist == NO_TWIST
            twist = NO_TWIST

        return cond_node, raise_node, twist


def get_dataset_loaders(path, tokenizer,
                        transform=None, batch_size=4, workers=4,
                        training_split=.8, shuffle=False, fraction=1.0,
                        random=False, ddp=False, eval_mode=False):
    """
    :param path: path to dataset
    :param tokenizer:
    :param transform:
    :param batch_size: batch size
    :param workers: # of threads for loading
    :param training_split: % of data used for validation [0, 1]
    :param shuffle: unsupported
    :param fraction: % of data used for training
    :param random: randomize choice of train/val/test split
    :param ddp: enable when multi-gpu training
    :param eval_mode:
    :return:
    """

    test_split = validation_split = (1 - training_split) / 2

    ds = IfRaisesDataset(path=path,
                         tokenizer=tokenizer,
                         transform=transform,
                         fraction=fraction,
                         eval_mode=eval_mode)

    train_len = int(training_split * len(ds))
    val_len = int(validation_split * len(ds))
    print(f'Train: {train_len}, Val: {val_len}, Test: {len(ds) - train_len - val_len}')

    if not random:
        train_ds, val_ds, test_ds = random_split(ds, [train_len, val_len, len(ds) - train_len - val_len],
                                                 generator=torch.Generator().manual_seed(42))
    else:
        train_ds = Subset(ds, np.arange(0, train_len))
        val_ds = Subset(ds, np.arange(train_len, train_len + val_len))
        test_ds = Subset(ds, np.arange(train_len + val_len, len(ds)))

    train_sampler = DistributedSampler(train_ds) if ddp else None
    val_sampler = DistributedSampler(val_ds) if ddp else None
    test_sampler = DistributedSampler(test_ds) if ddp else None

    if shuffle:
        pass
        # shuffling currently not working, ignore this
        # train_sampler = RandomSampler(train_ds)
        # val_sampler = RandomSampler(val_ds)

    loader_kwargs = {'batch_size': batch_size, 'num_workers': workers, 'pin_memory': True, 'drop_last': True}
    train_loader = DataLoader(train_ds, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, sampler=val_sampler, **loader_kwargs)
    test_loader = DataLoader(test_ds, sampler=test_sampler, **loader_kwargs)

    if batch_size is not None and training_split < 1.0:
        if len(train_loader.dataset.indices) < batch_size:
            raise UserWarning('Training data subset too small', len(train_loader.dataset.indices))

        if len(val_loader.dataset.indices) < batch_size:
            raise UserWarning('Validation data subset too small', len(val_loader.dataset.indices))

        if len(test_loader.dataset.indices) < batch_size:
            raise UserWarning('Test data subset too small', len(test_loader.dataset.indices))

    return train_loader, val_loader, test_loader
