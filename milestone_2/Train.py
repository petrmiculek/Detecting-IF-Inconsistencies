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
from src.eval import compute_metrics, accuracy

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', help="Path to functions file used for training.", required=True)
parser.add_argument(
    '--destination', help="Path to save your trained model.", required=True)

"""
evaluation:
    - accuracy #DONE#
    - confusion matrix #DONE#

data:
    - 
    

model todo:
    
    (once basic training works)
    - add bidirectional lstm
    - batch size
    - amp scaler - 16bit training
    - early stopping
    - lr scheduler/decay
    - add dropout
        
"""


"""
def train_model(model, source):
    pass


def save_model(model, destination):
    # Save model to destination.
    torch.save(model.state_dict(), destination)
"""


class ModelTraining:
    def __init__(self, dataset_path, ds_fraction=0.1):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        """ Dataset and Preprocessing """
        self.dataset_path = 'shared_resources/dataset_preprocessed_1000.pkl'
        # dataset_path = config.dataset_preprocessed_path
        self.dataset_path = dataset_path

        self.tokenizer = load_tokenizer(model_input_len=None)  # config.model_input_len

        self.dataset_fraction = ds_fraction
        datasets = get_dataset_loaders(dataset_path, self.tokenizer,
                                       fraction=self.dataset_fraction,
                                       batch_size=1, training_split=0.8,
                                       )
        self.train_dataset, self.val_dataset, self.test_dataset = datasets

        """ Model """
        # todo: put hyperparameters to config
        self.lr = 1e-3
        self.batch_size = 1

        self.init_model()

        self.print_setup()

        """ Logging """
        self.training_losses = []

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
        wb.watch(self.model, criterion=self.bce_loss, log="all", log_freq=1000)

    def init_model(self):
        # setting self-variables outside init for clarity and reusability
        self.epochs_trained = 0
        self.model = LSTM(config.embedding_dim, config.model['hidden_size'])
        self.model.cuda()
        self.bce_loss = nn.BCELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

    def print_setup(self):
        print(self.model)
        count_parameters(self.model)
        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    def train_model(self, epochs=20):
        start_time = time.time()
        print('Training starts...')

        # logging
        epoch_training_loss = np.nan
        epoch_validation_loss = np.nan
        predictions_train = []
        gts_train = []
        predictions_valid = []
        gts_valid = []


        # training + validation loop
        from_ = self.epochs_trained
        to_ = self.epochs_trained + epochs
        for epoch in range(from_, to_):
            epoch_training_losses = []
            epoch_validation_losses = []

            self.chosen_sample = next(iter(self.train_dataset))
            print(f'Training on sample {self.chosen_sample[1].item()}')

            self.model.train()
            # sample_train = self.train_dataset[0]
            """
            todo:
            
            2 samples fixed
            1 sample re-extracted
            10 samples fixed
            
            ditch the padding?
            
            """

            # one epoch
            # with tqdm.tqdm(enumerate(self.train_dataset)) as pbar:
            try:
                with tqdm.tqdm(range(10)) as pbar:
                    # for i, sample in pbar:
                    for i in pbar:
                        x, y = self.chosen_sample
                        gts_train.append(y.item())

                        x = x.to(device=self.device, dtype=torch.float)
                        y = y.to(device=self.device, dtype=torch.float)
                        self.model.zero_grad()
                        pred = self.model(x)
                        loss = self.bce_loss(pred[0], y)  # batch size 1
                        loss.backward()
                        self.optimizer.step()

                        # logging
                        with torch.no_grad():
                            loss_value = loss.item()
                            epoch_training_losses.append(loss_value)
                            self.training_losses.append(loss_value)
                            predictions_train.append(pred[0].item())
                            pbar.set_postfix(loss=f'{loss_value:.4f}')

            except KeyboardInterrupt:
                print('Stopped training through KeyboardInterrupt')

            self.model.eval()

            with torch.no_grad():
                with tqdm.tqdm(enumerate(self.val_dataset)) as pbar:
                    for i, sample in pbar:
                        x, y = sample
                        gts_valid.append(y)

                        x = x.to(device=self.device, dtype=torch.float)
                        y = y.to(device=self.device, dtype=torch.float)
                        pred = self.model(x)
                        loss = self.bce_loss(pred[0], y)

                        # logging
                        loss_value = loss.item()
                        epoch_validation_losses.append(loss_value)
                        predictions_valid.append(pred[0].item())

                        pbar.set_postfix(loss=f'{loss_value:.4f}')

            # logging
            epoch_training_loss = np.nanmean(epoch_training_losses)
            epoch_validation_loss = np.nanmean(epoch_validation_losses)
            accu_train = accuracy(gts_train, predictions_train)['accuracy']
            accu_valid = accuracy(gts_valid, predictions_valid)['accuracy']

            self.writer.add_scalars('Loss',
                                    {'train': epoch_training_loss,
                                     'val': epoch_validation_loss
                                     }, epoch)

            wb.log({'Train Loss': epoch_training_loss,
                    'Valid Loss': epoch_validation_loss,
                    'Train Acc': accu_train,
                    'Valid Acc': accu_valid,
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
        print(f'Model saved to "{path}"')


# def main():
if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Running on device: {device}')

    args = parser.parse_args()

    # con = 'shared_resources/real_test_for_milestone3/real_consistent.json'
    # incon = 'shared_resources/real_test_for_milestone3/real_inconsistent.json'
    # dataset_path = 'shared_resources/dataset_preprocessed_1000.pkl'

    model_training = ModelTraining(dataset_path=args.source)

    model_training.train_model(epochs=20)
    model_training.save_model(args.destination)

# if __name__ == '__main__':
#     main()
