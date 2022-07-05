#!/usr/bin/python3

# stdlib
import argparse
import datetime
import time
from copy import deepcopy as dc
import os
from os.path import join
import json
from typing import List, Tuple, Dict, Optional
import zipfile

"""
# currently unused
# fix for local import problems - add all local directories
import sys
sys_path_extension = [os.getcwd()]  # + [d for d in os.listdir() if os.path.isdir(d)]
sys.path.extend(sys_path_extension)
"""

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
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

# local
from src import config
from src.config import consistent, inconsistent
from src.preprocess import negate_cond, rebuild_cond, load_tokenizer
from src.util import count_parameters
from src.dataset import IfRaisesDataset
from src.model import LSTMBase as LSTM

# from ..milestone_1.run import model_predict

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', help="Path to functions file used for training.", required=True)
parser.add_argument(
    '--destination', help="Path to save your trained model.", required=True)

"""
data:
    - use embeddings fasttext
    - 
    

model todo:
    - TensorBoard
    - logging per epoch
    - batch size
    - lr scheduler/decay
    - early stopping
    - amp scaler
    - add dropout
    - add bidirectional lstm
        
"""


def model_predict(x):
    """
    interface dummy
    """
    result_dict = np.zeros(len(x))
    result_dict = [{k: v for (k, v) in zip(x['line_raise'], result_dict)}]
    return result_dict


def train_model(model, source):
    """
    TODO: Implement your method for training the model here.
    """

    raise Exception("Method not yet implemented.")


def save_model(model, destination):
    """
    Save model to destination.
    """
    torch.save(model.state_dict(), destination)


def print_setup(model, device):
    print(model)
    count_parameters(model)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')


# def main():
if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Running on device: {device}')

    args = parser.parse_args()
    # model = train_model(model=None, source=args.source)

    # save_model(model, args.destination)
    if False:
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

    # Model
    model = LSTM(config.embedding_dim, 64, 1)
    model.cuda()
    print_setup(model, device)

    dataset_path = '../shared_resources/dataset_preprocessed_1000.pkl'

    tokenizer = load_tokenizer(config.model_input_len)
    dataset = IfRaisesDataset(dataset_path, tokenizer=tokenizer)

    if False:
        lengths = []
        x = []
        y = []

        sep_token_id = tokenizer.token_to_id("</s>")
        sep_token_str = tokenizer.id_to_token(sep_token_id)
        # or just place '</s>' in the target list

        df = pd.read_pickle(dataset_path)
        for i, sample in df.iterrows():
            cond = sample['cond']
            # fix cond
            if sample['else']:
                cond = negate_cond(cond)
            else:
                cond = rebuild_cond(sample['cond'], parentheses=False)

            # get str repr
            cond_str = cst.Module([cond]).code
            raise_str = cst.Module([sample['raise']]).code

            # s['context'] + sep_token_str +
            full_str = " ".join([cond_str, sep_token_str, raise_str])

            tokens_cond = tokenizer.encode(cond_str)
            tokens_raise = tokenizer.encode(raise_str)
            tokens_context = tokenizer.encode(sample['context'])

            tokens_all = tokenizer.encode(full_str)

            lengths.append({
                'cond': real_len(tokens_cond),
                'raise': real_len(tokens_raise),
                'context': real_len(tokens_context),
                'all': real_len(tokens_all)
            })

            # tokens_both = tokens_cond.tokens + tokens_raise.tokens[1:]  # skip start token for 'sentence 2'
            # embeds = embed_model.wv[tokens_both]
            # embeddings.append(token_embeds)
            # embeds_raise = embed_model.wv[tokens_raise.tokens]

            target = consistent

            # x.append(embeds)
            # y.append(target)

    if True:
        # TensorBoard Logging
        training_run_id = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
        print(f'Training run ID: {training_run_id}')
        outputs_dir = join('runs', training_run_id)
        os.makedirs(outputs_dir, exist_ok=True)
        writer = SummaryWriter(outputs_dir)
        # writer.add_text('run_id', training_run_id)

        checkpoint_path = join(outputs_dir, f'checkpoint_{training_run_id}.pt')
        # early_stopping = EarlyStopping(patience=config.HYPERPARAMETERS['early_stopping_patience'],
        #                                     verbose=True, path=checkpoint_path)

    start_time = time.time()
    print('Training starts...')

    if True:
        # epoch = 0
        training_losses = []
        bce_loss = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        # split_idx = int(0.8 * len(x))
        for epoch in [0, 1]:
            # one epoch
            for i, sample in tqdm.tqdm(enumerate(dataset)):
                with torch.no_grad():
                    pass

                x, y = sample
                x = torch.tensor(x, device=device)
                y = torch.tensor([y], device=device)

                # print(model(x))
                model.zero_grad()
                pred = model(x[None, ...])
                loss = bce_loss(pred[0], y)
                loss.backward()
                optimizer.step()

                # logging
                mean_training_loss = loss.item()
                training_losses.append(mean_training_loss)

            mean_training_loss = np.mean(training_losses)

            writer.add_scalars('Loss/Total', {'train': mean_training_loss,
                                              # 'val': mean_validation_loss
                                              }, epoch)

    writer.flush()
    writer.close()

    time_elapsed = time.time() - start_time

    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.')

    save_model(model, config.model_weights_path)
