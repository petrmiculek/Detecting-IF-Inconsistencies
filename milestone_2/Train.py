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

import wandb as wb

# local
from src import config
from src.config import consistent, inconsistent
from src.extract import load_segments, extract_raises
from src.preprocess import negate_cond, rebuild_cond, load_tokenizer
from src.util import count_parameters
from src.dataset import IfRaisesDataset, get_dataset_loaders
from src.model import LSTMBase as LSTM

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', help="Path to functions file used for training.", required=True)
parser.add_argument(
    '--destination', help="Path to save your trained model.", required=True)

"""
evaluation:
    - accuracy
    - confusion matrix

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


"""
def train_model(model, source):
    pass


def s ave_model(model, destination):
    # Save model to destination.
    torch.save(model.state_dict(), destination)
"""


class ModelTraining:
    def __init__(self, dataset_path):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        """ Dataset and Preprocessing """
        self.dataset_path = '../shared_resources/dataset_preprocessed_1000.pkl'
        # dataset_path = config.dataset_preprocessed_path
        self.dataset_path = dataset_path

        self.tokenizer = load_tokenizer(config.model_input_len)
        # dataset = IfRaisesDataset(dataset_path, tokenizer=tokenizer, fraction=0.1, )

        self.dataset_fraction = 1.0
        datasets = get_dataset_loaders(dataset_path, self.tokenizer,
                                       fraction=self.dataset_fraction,
                                       batch_size=1
                                       )
        self.train_dataset, self.val_dataset, self.test_dataset = datasets

        """ Model """
        # todo: put hyperparameters to config
        self.lr = 1e-2
        self.batch_size = 1

        self.epochs_trained = 0

        self.model = LSTM(config.embedding_dim, 128, 1)
        self.model.cuda()
        self.bce_loss = nn.BCELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        self.print_setup()

        """ Logging """
        wb.init(project="asdl")

        self.training_run_id = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
        print(f'Training run ID: {self.training_run_id}')
        self.outputs_dir = join('runs', self.training_run_id)
        os.makedirs(self.outputs_dir, exist_ok=True)
        self.writer = SummaryWriter(self.outputs_dir)
        self.writer.add_text('run_id', self.training_run_id)

        self.checkpoint_path = join(self.outputs_dir, f'checkpoint_{self.training_run_id}.pt')
        # self.early_stopping = EarlyStopping(patience=config.HYPERPARAMETERS['early_stopping_patience'],
        #                                     verbose=True, path=self.checkpoint_path)

        wb.config = {
            "run_id": self.training_run_id,
            "learning_rate": self.lr,
            "batch_size": self.batch_size,
            "optimizer": str(self.optimizer),
            "dataset_fraction": self.dataset_fraction,
        }
        wb.watch(self.model, criterion=self.bce_loss, log="all", log_freq=100)

    def print_setup(self):
        print(self.model)
        count_parameters(self.model)
        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    def train(self, epochs=20):
        start_time = time.time()
        print('Training starts...')

        epoch_training_loss = np.nan
        epoch_validation_loss = np.nan

        from_ = self.epochs_trained
        to_ = self.epochs_trained + epochs
        for epoch in range(from_, to_):
            epoch_training_losses = []
            epoch_validation_losses = []

            self.model.train()
            # one epoch
            for i, sample in tqdm.tqdm(enumerate(self.train_dataset)):
                x, y = sample
                x = x.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.float)
                self.model.zero_grad()
                pred = self.model(x)
                loss = self.bce_loss(pred[0], y)  # batch size 1
                loss.backward()
                self.optimizer.step()

                # logging
                loss_value = loss.item()
                epoch_training_losses.append(loss_value)

            self.model.eval()

            for i, sample in tqdm.tqdm(enumerate(self.val_dataset)):
                with torch.no_grad():
                    x, y = sample
                    x = x.to(device=self.device, dtype=torch.float)
                    y = y.to(device=self.device, dtype=torch.float)
                    pred = self.model(x)
                    loss = self.bce_loss(pred[0], y)

                    # logging
                    loss_value = loss.item()
                    epoch_validation_losses.append(loss_value)

            # logging
            epoch_training_loss = np.nanmean(epoch_training_losses)
            epoch_validation_loss = np.nanmean(epoch_validation_losses)
            self.writer.add_scalars('Loss/Total',
                                    {'train': epoch_training_loss,
                                     'val': epoch_validation_loss
                                     }, epoch)
            wb.log({'loss_train': epoch_training_loss,
                    'loss_val': epoch_validation_loss
                    }, step=epoch)

        self.epochs_trained += epochs

        self.writer.flush()
        self.writer.close()

        time_elapsed = time.time() - start_time

        print(f'Training finished in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.')
        print(f'Trained epochs: {self.epochs_trained}\n'
              f'{epoch_training_loss=:.3f}\n'
              f'{epoch_validation_loss=:.3f}')

    def save_model(self, path):
        """
        Save model to path
        """
        torch.save(self.model.state_dict(), path)


# if __name__ == "__main__":
def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Running on device: {device}')

    args = parser.parse_args()

    # model = train_model(model=None, source=args.source)
    # save_model(model, args.destination)

    args.source = '../shared_resources/real_test_for_milestone3/real_consistent.json'
    source = '../shared_resources/real_test_for_milestone3/real_consistent.json'
    dataset_path = source

    # dataset_path = '../shared_resources/dataset_preprocessed_1000.pkl'
    model_training = ModelTraining(dataset_path=dataset_path)
    model_training.train(epochs=2)
    model_training.save_model(config.model_weights_path)

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


if __name__ == '__main__':
    main()
