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
from libcst.metadata.position_provider import PositionProvider
from libcst.metadata.parent_node_provider import ParentNodeProvider
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


# debugging functions

def model_predict(input):
    """
    interface dummy
    """
    line_numbers = [s['line_raise'] for s in input]
    results = np.zeros_like(line_numbers)
    return [{k: v for (k, v) in zip(line_numbers, results)}]


def r(node):
    if type(node) == list:
        return "".join(list(map(r, node)))
    else:
        return cst.Module([node]).code


def p(node):
    print(r(node))


q = 0


def pird(if_raise_dict, init=None):
    global q
    print(f"{q}: {rt(if_raise_dict['cond'], if_raise_dict['raise'])}")
    q += 1


def rt(cond, statement):
    return "if " + r(cond) + ":\n\t" + r(statement)


def n():
    global segments
    i = 0
    while True:
        print(f'=========='
              f'{i}:'
              f'{segments[i]}\n')
        i += 1
        input('')


def numbered_lines(string):
    for i, line in enumerate(string.split('\n')):
        print(f'{i:02d}: {line}')


def get_raise(body):
    for stmt in body:
        if type(stmt) == cst.Raise:
            return stmt


class FindIf(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider, ParentNodeProvider)

    def __init__(self):
        super().__init__()
        self.ifs = []
        self.current_line = 1
        self.lines = [None]

    def visit_If(self, node: cst.If):
        # p(node)
        line = self.get_metadata(PositionProvider, node).start.line
        self.ifs.append(node)
        self.current_line = line
        self.lines.append(line)

    def leave_If(self, original_node) -> None:
        self.lines.pop()


class FindRaise(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider, ParentNodeProvider)

    def __init__(self):
        super().__init__()
        self.raises = []
        self.lines = []
        self.samples = []

    def visit_Raise(self, node: cst.Raise):
        self.raises.append(node)
        # a = 5
        line = self.get_metadata(PositionProvider, node).start.line
        new_node = node
        else_branch = False
        try:
            while True:
                new_node = self.get_metadata(ParentNodeProvider, new_node)
                if type(new_node) == cst.Else:
                    else_branch = True
                    new_node = self.get_metadata(ParentNodeProvider, new_node)
                    # fall-through
                if type(new_node) == cst.If:
                    break

                if type(new_node) in [cst.Try, cst.TryStar, cst.ExceptHandler, cst.ExceptStarHandler]:
                    # ignore try-except-finally blocks
                    new_node = None
                    break

        except KeyError as e:
            # no containing `if` for this `raise`
            # print(f'{line=}')
            # print(f'<{r(node)}>'
            #       f'[{r(new_node)}]')
            # print("=" * 10)
            new_node = None

        if new_node is not None:
            try:
                context = self.get_metadata(ParentNodeProvider, new_node)
            except:
                context = None
            line_if = self.get_metadata(PositionProvider, new_node).start.line
            self.samples.append({'line_raise': line,
                                 'line_if': line_if,
                                 'raise': node,
                                 'cond': new_node.test,
                                 'else': else_branch,
                                 'context': None,
                                 })

        self.lines.append(line)


""" Unused - elif is an if + else """
"""
class FindElif(cst.CSTVisitor):
    def __init__(self):
        super().__init__()
        self.ifs = []

    def visit_Elif(self, node: cst.Elif):
        try:
            self.ifs.append(node)
        except Exception as e:
            print(e)
"""

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

"""
def get_raise_snippets(if_finder):
    snippets = []

    raise_finder = FindRaise()  # init only once

    for i, if_i in enumerate(if_finder.ifs):
        code_block = r(if_i)
        # body_ = body
        body = cst.MetadataWrapper(cst.parse_module(code_block))
        body.visit(raise_finder)

        # lines_whole = code_block.count('\n')
        lines_true = r(if_i.body).count('\n')
        # if len(raise_finder.raises) > 0:  # should loop through all found raises (in x-block?)

        negative_test = cst.UnaryOperation(operator=cst.Not(), expression=if_i.test)
        else_branch = False
        for (line_i, raise_i) in zip(raise_finder.lines, raise_finder.raises):
            if line_i > lines_true:
                else_branch = True

            test = negative_test if else_branch else if_i.test

            snippets.append(
                {
                    "if": test,
                    "raise": raise_i,
                    "line_if": if_finder.current_line,
                    "line_raise": line_i,
                    "code": code_block,
                    "active_if": if_finder.lines[-1]
                })

    return snippets
"""


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


segments_idx = {
    "simple_if": 1,

}


def inverse_operator(operator):
    if type(operator) == cst.Equal:
        return cst.NotEqual
    elif type(operator) == cst.NotEqual:
        return cst.Equal
    elif type(operator) == cst.GreaterThan:
        return cst.LessThanEqual
    elif type(operator) == cst.LessThan:
        return cst.GreaterThanEqual
    elif type(operator) == cst.GreaterThanEqual:
        return cst.LessThan
    elif type(operator) == cst.LessThanEqual:
        return cst.GreaterThan
    elif type(operator) == cst.Is:
        return cst.IsNot
    elif type(operator) == cst.IsNot:
        return cst.Is
    elif type(operator) == cst.In:
        return cst.NotIn
    elif type(operator) == cst.NotIn:
        return cst.In
    else:
        raise Exception(f"Unknown operator: {operator}")


def negate_cond(cond):
    try:
        if type(cond) == cst.UnaryOperation and type(cond.operator) == cst.Not:
            return cond.expression
        elif type(cond) == cst.Comparison:
            inv_op = inverse_operator(cond.comparisons[0].operator)
            tgt = cst.ComparisonTarget(operator=inv_op(), comparator=cond.comparisons[0].comparator)

            return cst.Comparison(
                left=cond.left,
                comparisons=[
                    tgt
                ]
            )
        # elif type(cond) == cst.And:
        # todo add DeMorgan's laws

        else:
            return cst.UnaryOperation(operator=cst.Not(), expression=cond)
    except Exception as e:
        print(e)
        return None


class LogTime:
    def __init__(self):
        self.start = time.time()
        self.times = []
        self.messages = []

    def log(self, msg):
        self.times.append(time.perf_counter())
        self.messages.append(msg)

    def print(self):
        print('='*10, '\n'
              ' i| time| message')
        for i, (t, msg) in enumerate(zip(self.times, self.messages)):
            print(f'{i:02d}| {(t - self.times[0]):2.2f}| {msg}')


if __name__ == '__main__':
    ts = LogTime()
    ts.log('start')
    archive_file = '../shared_resources/data.zip'
    file = 'functions_list.json'
    with zipfile.ZipFile(archive_file) as archive:
        with archive.open(file) as f:
            segments = json.load(f)

    ts.log('json read finished')
    # print(sum(map(g, segments)))

    samples = []
    for i, segment in enumerate(segments[45:47]):  # [74:75] # 208:209
        tree = cst.MetadataWrapper(cst.parse_module(segment))
        raise_finder = FindRaise()  # init only once
        tree.visit(raise_finder)
        for s in raise_finder.samples:
            # keep preceding context - lines before the `if`
            context_end_line = s['line_if'] - 1
            pre_context = "\n".join(segment.split('\n')[:context_end_line])
            s['context'] = pre_context

        samples.extend(raise_finder.samples)

    ts.log('samples extracted')

    for s in samples:
        not_cond = negate_cond(s['cond'])
        print("=" * 4)
        p(s['cond'])
        p(not_cond)

    ts.log('negated')
    """
    if_i = None
    # look for if-raise
    snippets = []
    for i, segment in enumerate(segments[8:9]):  # [74:75] # 208:209
        tree_if = cst.MetadataWrapper(cst.parse_module(segment))

        if_finder = FindIf()
        tree_if.visit(if_finder)
        snippets += get_raise_snippets(if_finder)

    ts.append(time.perf_counter())
    """

    # Exploration:

    # max_char_len = max(map(len, snippets))
    # print(f'Max observed input len = {max_char_len} chars.')

    df = pd.DataFrame(samples)

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

    # print(list(np.array(ts) - ts[0]))
    ts.print()
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
