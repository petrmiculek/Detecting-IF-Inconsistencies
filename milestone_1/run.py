# stdlib
import time
from copy import deepcopy as dc
import os
import json
from typing import List, Tuple, Dict, Optional
import zipfile

# external
import gensim
import libcst as cst
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import tokenizers

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

# local
import config


def r(node):
    if type(node) == list:
        return "".join(list(map(r, node)))
    else:
        return cst.Module([node]).code


def p(node):
    print(r(node))


def rt(cond, statement):
    return "if " + r(cond) + ":\n\t" + r(statement)


def get_raise(body):
    for stmt in body:
        if type(stmt) == cst.Raise:
            return stmt


class FindRaise(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.PositionProvider,)

    def __init__(self):
        super().__init__()
        self.raises = []
        self.lines = []

    def visit_Raise(self, node: cst.Raise):
        self.raises.append(node)
        line = self.get_metadata(cst.PositionProvider, node).start.line
        self.lines.append(line)


class FindIf(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.PositionProvider,)

    def __init__(self):
        super().__init__()
        self.ifs = []
        self.if_lines = []

    def visit_If(self, node: cst.If):
        line = self.get_metadata(cst.PositionProvider, node).start.line
        self.ifs.append(node)
        self.if_lines.append(line)


""" Unused - elif is an if + else """
# class FindElif(cst.CSTVisitor):
#     def __init__(self):
#         super().__init__()
#         self.ifs = []
#
#     def visit_Elif(self, node: cst.Elif):
#         try:
#             self.ifs.append(node)
#         except Exception as e:
#             print(e)

"""
class ChangeIf(cst.CSTTransformer):
    # todo unused
    def __init__(self):
        super().__init__()
        self.ifs = []
        self.other_calls = []
        self.prints_detailed = []
        self.stack: List[Tuple[str, ...]] = []

    def visit_If(self, node: cst.If):
        try:
            # node.body
            # node.orelse
            self.ifs.append(node)
        except Exception as e:
            print(e)

    def leave_If(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
        new_conditional = cst.Not(cst.If.test)
        new_node = updated_node.with_changes(test=new_conditional)
        return new_node
"""


def get_raise_snippets(if_finder, else_branch=False):
    snippets = []
    raises_all = []
    # raise_if_idx = []
    if_stack = []

    raise_finder = FindRaise()  # init only once

    for i, if_i in enumerate(if_finder.ifs):
        # true-block

        if else_branch:
            if if_i.orelse is None:
                continue
            body = if_i.orelse
        else:
            body = if_i.body

        body.visit(raise_finder)

        if len(raise_finder.raises) > 0:
            raises_all.append(raise_finder.raises)
            # raise_if_idx.append(i)
            # code = cst.Module([body]).code
            # cnt = code.count('raise')
            # if cnt == 1:
            if True:
                test = cst.Not(if_i.test) if else_branch else if_i.test
                # snippet = rt(test, raise_finder.raises[0])
                snippets.append((test, raise_finder.raises[0]))

            # elif cnt > 1:
            #     pass
            #     print(f'---\n'
            #           f'({i=}: {len(raise_finder.raises)} raises)\n'
            #           f'{code}\n'
            #           f'# ^unused^\n---')
            # else nothing

    return snippets


def load_tokenizer():
    tokenizer = ByteLevelBPETokenizer(
        "../shared_resources/pretrained_tokenizer/py_tokenizer-vocab.json",
        "../shared_resources/pretrained_tokenizer/py_tokenizer-merges.txt",
    )
    tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=config.model_input_len)
    tokenizer.enable_padding(length=config.model_input_len)
    return tokenizer


if __name__ == '__main__':
    # print('ASDL')
    ts = []
    ts.append(time.perf_counter())
    archive_file = '../shared_resources/data.zip'
    file = 'functions_list.json'
    with zipfile.ZipFile(archive_file) as archive:
        with archive.open(file) as f:
            segments = json.load(f)

    # print(sum(map(g, segments)))

    if_i = None
    # look for if-raise
    snippets = []
    for i, segment in enumerate(segments[70:90]):  # [74:75] #
        tree_if = cst.parse_module(segment)

        if_finder = FindIf()
        tree_if.visit(if_finder)
        snippets += get_raise_snippets(if_finder, else_branch=False)
        snippets += get_raise_snippets(if_finder, else_branch=True)

    ts.append(time.perf_counter())

    # Exploration:
    # max_char_len = max(map(len, snippets))
    # print(f'Max observed input len = {max_char_len} chars.')

    if False:
        tokenizer = load_tokenizer()
        embed_model = gensim.models.FastText.load(config.fasttext_path)

        embeddings = []
        token_lens = []
        for s in snippets:
            tokens = tokenizer.encode(s)
            token_lens.append(tokens.tokens.index('[PAD]'))
            token_embeds = embed_model.wv[tokens.tokens]
            embeddings.append(token_embeds)

        print(f'Max observed input len = {max(token_lens)} tokens.')

        np_embs = np.stack(embeddings, axis=0)
        # -> [Batch size, Length, Dimension]
        ts.append(time.perf_counter())

        np.save(config.dataset_tokenized_path, np_embs)

        print(list(np.array(ts) - ts[0]))










































    """
    Where I finished:
        .    
    Now:    
        How come I don't get nested ifs?
        - even when only looking for a raise
          inside the if-body we could still find
          one that's deeper down
        - todo: run visitor on nested ifs example
        
        
        if.orelse can be If ( = elif) or Else ( = else)
        

    What I need to do:
        - elif
            it will always be in inside a if body
            at a local scope, it's just an if
        - save embeddings 
            DONE
        - run for whole dataset
        - if a condition is not-X, then negated version must be X (currently not-not-X)
    
    
    - nech si odděleně ještě i teď textovou podobu - budeš nad tím dělat úpravy pro inkonzistentní vstupy
    - dodělat elify
    - kolik nechat z okolního kontextu (předchozí unrelated příkazy mohou a nemusí být důležité)
    - podmínka může pracovat s proměnnými, které jsou mimo scope (dodatečně dodat informaci o nich?)
    - nechat pouze část podmínky?
- 
    """
