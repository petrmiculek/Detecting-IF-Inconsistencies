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

import wandb

# local
from src import config
from src.config import consistent, inconsistent
from src.preprocess import negate_cond, rebuild_cond, load_tokenizer
from src.util import count_parameters
from src.dataset import IfRaisesDataset, get_dataset_loaders
from src.model import LSTMBase as LSTM

# from ..milestone_1.run import model_predict

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', help="Path to functions file used for training.", required=True)
parser.add_argument(
    '--destination', help="Path to save your trained model.", required=True)

"""
data:
    - 
    

model todo:
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


from src.extract import load_segments, extract_raises

# def main():
if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Running on device: {device}')

    args = parser.parse_args()
    # model = train_model(model=None, source=args.source)
    # save_model(model, args.destination)

    # segments = load_segments(archive=None, file=args.source)
    # samples = extract_raises(segments, max=None)


    if False:
        # take inspiration from TrainingArguments and Trainer
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

    """ Dataset and Preprocessing """
    dataset_path = '../shared_resources/dataset_preprocessed_1000.pkl'
    # dataset_path = config.dataset_preprocessed_path
    tokenizer = load_tokenizer(config.model_input_len)
    dataset = IfRaisesDataset(dataset_path, tokenizer=tokenizer, fraction=0.1, )

    dataset_fraction = 0.1
    train_dataset, val_dataset, test_dataset = get_dataset_loaders(dataset_path, tokenizer,
                                                                   fraction=dataset_fraction,
                                                                   batch_size=1
                                                                   )

    """ Model """
    # todo: put to config
    lr = 1e-2
    epochs = 20
    batch_size = 1

    model = LSTM(config.embedding_dim, 128, 1)
    model.cuda()
    print_setup(model, device)

    bce_loss = nn.BCELoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr)


    if True:
        """ Logging """
        wandb.init(project="asdl")

        training_run_id = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
        print(f'Training run ID: {training_run_id}')
        outputs_dir = join('runs', training_run_id)
        os.makedirs(outputs_dir, exist_ok=True)
        writer = SummaryWriter(outputs_dir)
        writer.add_text('run_id', training_run_id)

        checkpoint_path = join(outputs_dir, f'checkpoint_{training_run_id}.pt')
        # early_stopping = EarlyStopping(patience=config.HYPERPARAMETERS['early_stopping_patience'],
        #                                     verbose=True, path=checkpoint_path)

        wandb.config = {
            "run_id": training_run_id,
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": str(optimizer),
            "dataset_fraction": dataset_fraction,
        }
        wandb.watch(model, criterion=bce_loss, log="all", log_freq=100)

    start_time = time.time()
    print('Training starts...')

    epochs_trained = 0

    for epoch in range(epochs_trained, epochs_trained + epochs):
        epoch_training_losses = []
        epoch_validation_losses = []

        model.train()
        # one epoch
        for i, sample in tqdm.tqdm(enumerate(train_dataset)):
            x, y = sample
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)
            model.zero_grad()
            pred = model(x)
            loss = bce_loss(pred[0], y)  # batch size 1
            loss.backward()
            optimizer.step()

            # logging
            loss_value = loss.item()
            epoch_training_losses.append(loss_value)

        model.eval()

        for i, sample in tqdm.tqdm(enumerate(val_dataset)):
            with torch.no_grad():
                x, y = sample
                x = x.to(device=device, dtype=torch.float)
                y = y.to(device=device, dtype=torch.float)
                pred = model(x)
                loss = bce_loss(pred[0], y)

                # logging
                loss_value = loss.item()
                epoch_validation_losses.append(loss_value)

        epoch_training_loss = np.nanmean(epoch_training_losses)
        epoch_validation_loss = np.nanmean(epoch_validation_losses)
        writer.add_scalars('Loss/Total',
                           {'train': epoch_training_loss,
                            'val': epoch_validation_loss
                            }, epoch)
        wandb.log({'loss_train': epoch_training_loss,
                   'loss_val': epoch_validation_loss
                   }, step=epoch)

    writer.flush()
    writer.close()

    time_elapsed = time.time() - start_time

    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.')

    save_model(model, config.model_weights_path)
