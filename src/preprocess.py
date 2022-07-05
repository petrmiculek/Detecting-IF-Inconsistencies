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
        elif type(cond) in {cst.Attribute, cst.Call}:
            # don't add parentheses
            return cst.UnaryOperation(operator=cst.Not(), expression=cond)
        else:
            # add parentheses
            parenthesized = rebuild_cond(cond, parentheses=True)
            return cst.UnaryOperation(operator=cst.Not(), expression=parenthesized)

    except Exception as e:
        print(e)
        return None


def load_tokenizer(model_input_len=None):
    """
    Loads a tokenizer from a pre-trained model.
    :param model_input_len: int to pad, None to keep as is
    """
    tokenizer = ByteLevelBPETokenizer(
        "../shared_resources/pretrained_tokenizer/py_tokenizer-vocab.json",
        "../shared_resources/pretrained_tokenizer/py_tokenizer-merges.txt",
    )
    tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    # num_added_toks = tokenizer.add_tokens(['[SEP]'], special_tokens=True)  # his line is updated

    # model.resize_token_embeddings(len(tokenizer))

    # The tokenizer has to be saved if it has to be reused
    # tokenizer.save_pretrained(config.tokenizer_path)

    if model_input_len is not None:
        tokenizer.enable_truncation(max_length=model_input_len)
        tokenizer.enable_padding(length=model_input_len)
    return tokenizer

