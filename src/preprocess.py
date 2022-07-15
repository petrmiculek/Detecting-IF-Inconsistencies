import libcst as cst

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


def inverse_operator(operator):
    inverse_mapping = {
        cst.Equal: cst.NotEqual,
        cst.NotEqual: cst.Equal,
        cst.GreaterThan: cst.LessThanEqual,
        cst.LessThanEqual: cst.GreaterThan,
        cst.GreaterThanEqual: cst.LessThan,
        cst.LessThan: cst.GreaterThanEqual,
        cst.Is: cst.IsNot,
        cst.IsNot: cst.Is,
        cst.In: cst.NotIn,
        cst.NotIn: cst.In,

    }
    op_type = type(operator)
    if op_type in inverse_mapping:
        return inverse_mapping[op_type]
    else:
        raise Exception(f"Unknown operator: {operator}")


def rebuild_cond(cond, parentheses=False):
    # todo test for all possible types: BooleanOperation, BinaryOperation (5 + 5)
    kw = {k: cond.__getattribute__(k) for k in cond.__slots__}
    if parentheses:
        # wrap in parentheses
        kw['lpar'] = [cst.LeftParen(cst.SimpleWhitespace(value=''))]
        kw['rpar'] = [cst.RightParen(cst.SimpleWhitespace(value=''))]
    else:
        # remove parentheses
        kw['lpar'] = []
        kw['rpar'] = []
    rebuilt = type(cond)(**kw)
    return rebuilt


def negate_cond(cond):
    try:

        if type(cond) == cst.UnaryOperation and type(cond.operator) == cst.Not:
            # extract `x` from `not (x)`
            return rebuild_cond(cond.expression, parentheses=False)
        elif type(cond) == cst.Comparison:
            # parentheses removed
            inv_op = inverse_operator(cond.comparisons[0].operator)
            target = cst.ComparisonTarget(operator=inv_op(), comparator=cond.comparisons[0].comparator)

            return cst.Comparison(left=cond.left, comparisons=[target])
        # elif type(cond) == cst.And:
        # todo add DeMorgan's laws
        elif type(cond) in [cst.Attribute, cst.Call]:
            # don't add parentheses
            return cst.UnaryOperation(operator=cst.Not(), expression=cond)
        else:
            # add parentheses
            parenthesized = rebuild_cond(cond, parentheses=True)
            return cst.UnaryOperation(operator=cst.Not(), expression=parenthesized)

    except Exception as e:
        print(e)
        return None


def load_tokenizer(tokens_length=None):
    """
    Loads a tokenizer from a pre-trained model.
    :param tokens_length: int to pad, None to keep as is
    """
    vocab = "shared_resources/pretrained_tokenizer/py_tokenizer-vocab.json"
    merges = "shared_resources/pretrained_tokenizer/py_tokenizer-merges.txt"
    tokenizer = ByteLevelBPETokenizer(vocab, merges)

    tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )

    tokenizer.enable_padding(length=tokens_length)

    return tokenizer


def tokenize(tokenizer, text, max_len=None, truncate='right'):
    tokenized = tokenizer.encode(text).tokens

    if max_len is not None:
        if truncate == 'left':
            # keep last max_len tokens
            # not needed for shorter than max length, but in such case this doesn't do anything
            tokenized = tokenized[0:1] + tokenized[-(max_len - 1):]

        elif truncate == 'right':
            # keep first max_len tokens
            tokenized = tokenized[0:max_len]
        elif truncate == 'first_last':
            if len(tokenized) > max_len:
                tokenized = tokenized[0:max_len // 2] + tokenized[-(max_len // 2):]
        else:
            # do nothing
            pass

    return tokenized
