import libcst as cst
import numpy as np
import pandas as pd
import nltk
import tokenizers
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile

import os
import json

def p(node):
    print(cst.Module([node]).code)

class FindRaise(cst.CSTVisitor):
    # INIT
    def __init__(self):
        super().__init__()
        self.raises = []
        self.other_calls = []
        self.prints_detailed = []

    # name of the function should mention the type of node we are looking for, in this case a Call node
    def visit_Raise(self, node: cst.Raise):
        try:
            # a = 5
            # # cst.Module([node]).code
            #
            # node.test
            # node.body
            # node.orelse
            self.raises.append(node)
        except Exception as e:
            print(e)
            # print("CALL DOES NOT HAVE ATTR VALUE", cst.Module([node]).code)



class FindIf(cst.CSTVisitor):

    # data structures to keep track of the search results

    prints = []  # the list of print calls found in the code
    prints_detailed = []  # the details of the print call (arguments passed to the function print)
    other_calls = []  # other calls that are not print (this is just for the sake of the example, you can drop this attribute if you don't want that list)

    # INIT
    def __init__(self):
        super().__init__()
        self.ifs = []
        self.other_calls = []
        self.prints_detailed = []

    # name of the function should mention the type of node we are looking for, in this case a Call node
    def visit_If(self, node: cst.If):
        try:
            # a = 5
            # # cst.Module([node]).code
            #
            #
            # node.test
            # node.body
            # node.orelse
            self.ifs.append(node)
        except Exception as e:
            print(e)
            # print("CALL DOES NOT HAVE ATTR VALUE", cst.Module([node]).code)


if __name__ == '__main__':
    print('ASDL')

    archive_file = '../shared_resources/data.zip'
    file = 'functions_list.json'
    with zipfile.ZipFile(archive_file) as archive:
        with archive.open(file) as f:
            s = json.load(f)

    # look for if-raise

    # mock: first only
    e1 = s[0]
    tree_if = cst.parse_module(e1)

    ifs = FindIf()
    tree_if.visit(ifs)

    raises_all = []

    # if True:
    #     if_i = ifs.ifs[0]
    for if_i in ifs.ifs:
        # p(if_i)
        raises = FindRaise()
        if_i.visit(raises)
        raises_all.append(raises.raises)


    """
    Where I finished:
    
    there are no raise-s in the snippets found
    
    p(if_i) prints only one-line pieces of code
    Do I need to access `body` to get the whole body of the if?
    
    find one raise yourself
    
    open your live notes
    
    """

