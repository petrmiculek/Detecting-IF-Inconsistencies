# stdlib
import os
import sys

sys.path.append(os.path.join(os.getcwd(), os.pardir))

# external
import libcst as cst
import pandas as pd

# local
from src import config
from src.extract import extract_raises, load_segments
from src.preprocess import rebuild_cond, negate_cond, load_tokenizer
from src.util import LogTime, r, p, repr_cond_raise


# debugging functions

def print_all_segments(segments):
    i = 0
    while True:
        print(f'=========='
              f'{i}:'
              f'{segments[i]}\n')
        i += 1
        input('')


# interesting segment indices
segments_idx = {
    "simple_if": 1,
    "simple_else": 5,
    "if_elif_else_raise": 10,
    "if_if_raise": 11,
    "raise_outside_if": 2,
    "if_raise_else_if_raise": 8,
    "if_try_raise_except_raise": 28,
    "same1": 59,
    "same2": 68,
    "for_else_raise": 172,
    "except_if_raise_else_raise": 208,
}


def print_samples(samples):
    for s in samples:
        print("=" * 4)
        # print(s['context'])
        if s['else']:
            not_cond = negate_cond(s['cond'])
            p(not_cond)
        else:
            without_parentheses = rebuild_cond(s['cond'], parentheses=False)
            p(without_parentheses)
            # p(s['cond'])
        p(s['raise'])


# def main():
if __name__ == '__main__':
    ts = LogTime()
    ts.log('start')

    segments = load_segments()

    ts.log('json read finished')
    # print(sum(map(g, segments)))

    samples = extract_raises(segments, max=None)

    ts.log('samples extracted')

    # print_samples(samples)
    # ts.log('printed')

    # Exploration:
    # max_char_len = max(map(len, snippets))
    # print(f'Max observed input len = {max_char_len} chars.')

    df = pd.DataFrame(samples)
    pd.to_pickle(df, config.dataset_preprocessed_path)
    # dd = pd.read_pickle(config.dataset_preprocessed_path)

    if False:
        tokenizer = load_tokenizer(config.model_input_len)
        embed_model = gensim.models.FastText.load(config.fasttext_path)

        embeddings = []
        token_lens = []
        for s in samples:
            tokens = tokenizer.encode(s)
            token_lens.append(tokens.tokens.index('[PAD]'))
            token_embeds = embed_model.wv[tokens.tokens]
            embeddings.append(token_embeds)

        print(f'Max observed input len = {max(token_lens)} tokens.')

        np_embs = np.stack(embeddings, axis=0)
        # -> [Batch size, Length, Dimension]
        ts.append(time.perf_counter())

        np.save(config.dataset_tokenized_path, np_embs)

    # print(list(np.array(ts) - ts[0]))
    ts.print()

    count = 0
    for s in segments:
        c = s.count('raise')
        count += c

    """
    Where I finished:
        .    
    Now:
        
        if.orelse can be If ( = elif) or Else ( = else)
        

    What I need to do:
        - elif
            it will always be in inside a if body
            at a local scope, it's just an if
        - save embeddings 
            DONE
        - run for whole dataset
            DONE
        - dodělat elify
    
        - Conditions:

- 
    """

# if __name__ == '__main__':
#     main()
