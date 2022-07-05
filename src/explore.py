import os

import pandas as pd

from src.util import r, real_len
from src import config


def head(x, n=30):
    return x[:min(n, len(x))]


if __name__ == '__main__':
    df = pd.read_pickle('../data/dataset_preprocessed_full.pkl')

    print(f'{len(df)} samples in total')

    for attr in ['cond', 'raise', 'context']:
        if attr == 'context':
            strings = df[attr]
        else:
            strings = list(map(r, df[attr]))
        strings_lengths = list(map(len, strings))

        print(f'{attr}: '
              f'\tmin: {min(strings_lengths)}\n'
              f'\tmax: {max(strings_lengths)}\n'
              f'\tavg: {sum(strings_lengths) / len(strings_lengths)}')

        from src.preprocess import load_tokenizer

        tokenizer = load_tokenizer(config.model_input_len)
        tokenized = list(map(tokenizer.encode, strings))
        real_lens = list(map(real_len, tokenized))

        # ditto for tokenized
