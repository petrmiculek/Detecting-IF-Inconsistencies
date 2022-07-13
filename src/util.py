import time
from prettytable import PrettyTable
import libcst as cst


def r(node):
    # repr
    if type(node) == list:
        return "".join(list(map(r, node)))
    else:
        return cst.Module([node]).code


def p(node):
    # print
    print(r(node))


def repr_cond_raise(cond, statement):
    return "if " + r(cond) + ":\n\t" + r(statement)


def real_len(tokens):
    if tokens.attention_mask[-1] == 1:
        length = len(tokens.tokens)
    else:
        length = tokens.tokens.index('[PAD]')
    return length


class LogTime:
    def __init__(self):
        self.start = time.time()
        self.times = []
        self.messages = []

    def log(self, msg):
        self.times.append(time.perf_counter())
        self.messages.append(msg)

    def print(self):
        # flag: from_index, new_only
        print('=' * 16, '\n'
                        ' i| time| message')
        for i, (t, msg) in enumerate(zip(self.times, self.messages)):
            print(f'{i:02d}| {(t - self.times[0]):2.2f}| {msg}')


def count_parameters(model):
    """
    Counts total number of trainable parameters of given torch model. Prints table of its layers.

    :param model: torch model of NN

    :return: Number of trainable parameters.
    """

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    print(table)
    print(f"Total Trainable Params: {total_params}")

    return total_params


def get_dict(obj):
    return {key: value for key, value
            in obj.__dict__.items()
            if not key.startswith('_')}
