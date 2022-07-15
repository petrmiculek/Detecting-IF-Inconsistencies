import json
from zipfile import ZipFile

import libcst as cst
import tqdm
from libcst.metadata.parent_node_provider import ParentNodeProvider
from libcst.metadata.position_provider import PositionProvider


def load_segments(archive='shared_resources/data.zip', file='functions_list.json'):
    if archive is not None:
        with ZipFile(archive) as archive:
            with archive.open(file) as f:
                segments = json.load(f)
    else:
        with open(file) as f:
            segments = json.load(f)
    return segments


class FindRaise(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider, ParentNodeProvider)

    def __init__(self):
        super().__init__()
        self.raises = []
        self.lines = []
        self.samples = []

    def visit_Raise(self, node: cst.Raise):
        self.raises.append(node)
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
                    # ignore try-except-finally blocks (when there's no if inside them)
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
            # pre-context extracted from segment segment string form
            # try:
            #     context = self.get_metadata(ParentNodeProvider, new_node)
            # except:
            #     context = None
            line_if = self.get_metadata(PositionProvider, new_node).start.line

            self.samples.append({'line_raise': line,
                                 'line_if': line_if,
                                 'raise': node,
                                 'cond': new_node.test,
                                 'else': else_branch,
                                 'context': None,
                                 })

        self.lines.append(line)


def extract_raises(segments, max=None):
    samples = []
    start = 0
    # idx = list((segments_idx.values()))
    print(f'{len(segments)=}')
    if max is not None:
        print(f'len_used={max}')

    segments_used = segments[start:max]
    for i, segment in tqdm.tqdm(enumerate(segments_used, start=start)):  # [start: start + 2]  # [74:75] # 208:209
        tree = cst.MetadataWrapper(cst.parse_module(segment))
        raise_finder = FindRaise()  # init only once
        tree.visit(raise_finder)
        for s in raise_finder.samples:
            # keep preceding context - lines before the `if`
            context_end_line = s['line_if'] - 1
            pre_context = "\n".join(segment.split('\n')[:context_end_line])
            s['context'] = pre_context
            s['segment'] = start + i

        samples.extend(raise_finder.samples)
    return samples
