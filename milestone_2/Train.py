#!/usr/bin/python3

# stdlib
import argparse
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
# from ..milestone_1.run import model_predict

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', help="Path to functions file used for training.", required=True)
parser.add_argument(
    '--destination', help="Path to save your trained model.", required=True)




def train_model(model, source):
    """
    TODO: Implement your method for training the model here.
    """

    raise Exception("Method not yet implemented.")


def save_model(model, destination):
    """
    TODO: Implement your method for saving the training the model here.
    """
    raise Exception("Method not yet implemented.")


if __name__ == "__main__":
    args = parser.parse_args()
    # model = train_model(model=None, source=args.source)

    # save_model(model, args.destination)

    from transformers import AutoTokenizer
    from transformers import DataCollatorWithPadding
    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
    from datasets import load_dataset

    imdb = load_dataset("imdb")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_imdb = imdb.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny-mnli", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        # per_device_train_batch_size=16,
        # per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        # eval_dataset=tokenized_imdb["test"],
        tokenizer=tokenizer,
        # data_collator=data_collator,
    )

    trainer.train()
