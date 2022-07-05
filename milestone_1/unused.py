import libcst as cst
from libcst.metadata.position_provider import PositionProvider
from libcst.metadata.parent_node_provider import ParentNodeProvider

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
