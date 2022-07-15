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
from contextlib import suppress

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

"""
# currently unused
# fix for local import problems - add all local directories
import sys
sys_path_extension = [os.getcwd()]  # + [d for d in os.listdir() if os.path.isdir(d)]
sys.path.extend(sys_path_extension)
"""

# external
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import tokenizers
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb as wb
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# local
from src import config
from src.preprocess import negate_cond, rebuild_cond, load_tokenizer
from src.util import count_parameters, get_dict
from src.dataset import IfRaisesDataset, get_dataset_loaders
from src.model import LSTMBase
from src.eval import compute_metrics, accuracy, plot_roc_curve

from src.model_util import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', help="Path to functions file used for training.", required=True)
parser.add_argument(
    '--destination', help="Path to save your trained model.", required=True)

args_global = None
"""
evaluation:
    - accuracy #DONE#
    - confusion matrix #DONE#
    - eval test set #DONE#

data:
    - train on full dataset  #DONE#
    - if len(val_dataset) > 0: ... rejected
    
model todo:
    
    (once basic training works)
    - add bidirectional lstm  #DONE#
    - batch size  #DONE#
    - amp scaler - 16bit training  #DONE#
    - early stopping #DONE#
    - lr scheduler/decay  #DONE#
    - add dropout #DONE#
        
"""

"""
def train_model(model, source):
    pass


def save_model(model, destination):
    # Save model to destination.
    torch.save(model.state_dict(), destination)
"""


class ModelTraining:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        """ Dataset and Preprocessing """
        self.dataset_path = config.dataset_preprocessed_path

        self.tokenizer = load_tokenizer(tokens_length=config.tokens_length_unit)

        self.dataset_fraction = config.dataset_fraction
        self.batch_size = config.HPARS['batch_size']

        datasets = get_dataset_loaders(self.dataset_path, self.tokenizer,
                                       fraction=self.dataset_fraction,
                                       batch_size=self.batch_size, training_split=config.training_split,
                                       workers=4
                                       )
        self.train_dataset, self.val_dataset, self.test_dataset = datasets

        self.lr = config.HPARS['learning_rate']

        """ Model """
        self.init_model()

        """ Logging """
        self.epochs_stats = []

        self.training_run_id = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
        self.outputs_dir = join('runs', self.training_run_id)
        os.makedirs(self.outputs_dir, exist_ok=True)
        self.writer = SummaryWriter(self.outputs_dir)
        self.writer.add_text('run_id', self.training_run_id)

        self.checkpoint_path = join(self.outputs_dir, f'checkpoint_{self.training_run_id}.pt')
        self.early_stopping = EarlyStopping(patience=config.HPARS['early_stopping_patience'],
                                            verbose=True, path=self.checkpoint_path)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              verbose=True,
                                                              patience=config.HPARS['lr_scheduler_patience'],
                                                              min_lr=config.HPARS['lr_scheduler_min_lr'],
                                                              factor=config.HPARS['lr_scheduler_factor'])

        self.scaler = torch.cuda.amp.GradScaler()

        wb.config = {
            "run_id": self.training_run_id,
            "learning_rate": self.lr,
            "batch_size": self.batch_size,
            "optimizer": str(self.optimizer),
            "dataset_fraction": self.dataset_fraction,
        }

        config_dump = get_dict(config)
        wb.config.update(config_dump)
        global args_global
        args_dict = get_dict(args_global)
        wb.config.update(args_dict)
        wb.init(project="asdl", config=wb.config)

        # unused
        # wb.watch(self.model, criterion=self.bce_loss, log="all", log_freq=1000)

        print(f'Training run ID: {self.training_run_id}')
        self.print_setup()

        # real test datasets
        cons = "shared_resources/real_test_for_milestone3/real_consistent.json"
        incons = "shared_resources/real_test_for_milestone3/real_inconsistent.json"

        self.real_dataset_cons, _, _ = get_dataset_loaders(cons, self.tokenizer, fraction=1., eval_mode=True)
        self.real_dataset_incons, _, _ = get_dataset_loaders(incons, self.tokenizer, fraction=1., eval_mode=True)
        self.real_test_preds = []
        self.real_test_gts = []

    def init_model(self):
        # setting self-variables outside init for clarity and reusability
        self.epochs_trained = 0
        self.model = LSTMBase(config.embedding_dim, config.model['hidden_size'],
                              tokens_length=config.model_input_len)
        self.model.cuda()
        # self.bce_loss = nn.BCELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def pred_sample(self, sample):
        """
        Predict sample
        debugging
        """
        with torch.no_grad():
            x, y = sample  # self.chosen_sample
            x = x.to(device=self.device, dtype=torch.float)
            y = y.to(device=self.device, dtype=torch.float)
            pred = self.model.predict(x)
        return pred

    # debugging
    def pred_lstm(self, sample):
        with torch.no_grad():
            x, y = sample  # self.chosen_sample
            x = x.to(device=self.device, dtype=torch.float)
            y = y.to(device=self.device, dtype=torch.float)

            out, (h, c) = self.model.lstm(x)

            # h = h.view(h.size(0), -1)

            return h

    def eval_real_test(self):
        """
        Evaluate model on test set
        """
        if self.test_dataset is None:
            print('No test dataset found.')
            return
        print('Evaluating model on real data:')
        self.model.eval()
        self.model.test_mode = True

        def pred_inner(dataset):
            predictions = []
            for i, s in tqdm.tqdm(enumerate(dataset)):
                with torch.no_grad():
                    x, _, line = s
                    x = x.to(device=self.device)
                    pred = self.model.predict(x)
                    predictions.append(pred)

            preds = torch.cat(predictions, dim=0).cpu()
            self.real_test_preds.append(preds)
            return preds

        preds_cons = pred_inner(self.real_dataset_cons)
        preds_incons = pred_inner(self.real_dataset_incons)
        gts_cons = torch.zeros_like(preds_cons)
        gts_incons = torch.zeros_like(preds_cons) + 1.0

        gts_all = torch.cat([gts_cons, gts_incons]).cpu()
        preds_all = torch.cat([preds_cons, preds_incons]).cpu()
        plot_roc_curve(gts_all, preds_all, show=False, output_location='results')

        accuracy_consistent = accuracy(gts_cons, preds_cons)['accuracy']
        accuracy_inconsistent = accuracy(gts_incons, preds_incons)['accuracy']
        print(f'Accuracy consistent: {accuracy_consistent:.4f}')
        print(f'Accuracy inconsistent: {accuracy_inconsistent:.4f}')

        loss_cons = self.bce_loss(preds_cons, gts_cons).item()
        loss_incons = self.bce_loss(preds_incons, gts_incons).item()
        loss = 0.5 * (loss_cons + loss_incons)
        # print(f'Loss consistent: {loss_cons:.4f}')
        # print(f'Loss inconsistent: {loss_incons:.4f}')
        print(f'Loss: {loss:.4f}')
        self.model.test_mode = False
        return {'Accuracy Consistent Real Test': accuracy_consistent,
                'Accuracy Inconsistent Real Test': accuracy_inconsistent,
                'Loss Real Test': loss}

    def train_model(self, epochs=1):
        start_time = time.time()
        print('Training starts...')

        # training + validation loop
        from_ = self.epochs_trained
        to_ = self.epochs_trained + epochs
        try:
            for epoch in range(from_, to_):
                # logging
                epoch_training_losses = []
                epoch_validation_losses = []
                predictions_train = []
                gts_train = []
                predictions_valid = []
                gts_valid = []

                # self.chosen_sample = next(iter(self.train_dataset))
                # print(f'Training on sample {self.chosen_sample[1].item()}')

                self.model.train()
                # sample_train = self.train_dataset[0]

                # one epoch
                print(f'\nEpoch {epoch}:')
                with tqdm.tqdm(self.train_dataset, leave=False) as pbar:
                    i = 0
                    # for i, sample in pbar:
                    for s in pbar:
                        x, y = s  # self.chosen_sample
                        with torch.no_grad():
                            gts_train.append(y)

                        x = x.to(device=self.device, dtype=torch.float)
                        y = y.to(device=self.device, dtype=torch.float)

                        with torch.cuda.amp.autocast():
                            pred = self.model(x)[:, 0]
                            loss = self.bce_loss(pred, y)

                        self.scaler.scale(loss / 4).backward()  # Calculate the gradients

                        if (i + 1) % 4 == 0:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()  # Clear all the gradients before calculating them

                        # logging
                        with torch.no_grad():
                            loss_value = loss.item()
                            epoch_training_losses.append(loss_value)
                            predictions_train.append(pred)
                            pbar.set_postfix(loss=f'{loss_value:.4f}')

                        i += 1

                self.model.eval()

                with torch.no_grad():
                    with tqdm.tqdm(self.val_dataset, leave=False) as pbar:
                        for sample in pbar:
                            x, y = sample
                            gts_valid.append(y)

                            x = x.to(device=self.device, dtype=torch.float)
                            y = y.to(device=self.device, dtype=torch.float)
                            pred = self.model.predict(x)[:, 0]
                            loss = self.bce_loss(pred, y)

                            # logging
                            loss_value = loss.item()
                            epoch_validation_losses.append(loss_value)
                            predictions_valid.append(pred)

                            pbar.set_postfix(loss=f'{loss_value:.4f}')

                # logging
                loss_training = np.nanmean(epoch_training_losses)
                loss_validation = np.nanmean(epoch_validation_losses)
                accu_train = accuracy(gts_train, predictions_train)['accuracy']
                accu_valid = accuracy(gts_valid, predictions_valid)['accuracy']

                print('\nReal Test Data Results')  # newline
                epoch_real_test_results = self.eval_real_test()
                wb.log(epoch_real_test_results, step=epoch)

                res = {'Loss Training': loss_training,
                       'Loss Validation': loss_validation,
                       'Accuracy Training': accu_train,
                       'Accuracy Validation': accu_valid,
                       }

                print('')  # newline
                for k, v in res.items():
                    print(f'{k}: {v:.4f}')

                wb.log(res, step=epoch)
                self.epochs_stats.append(res)
                self.writer.add_scalars('Loss',
                                        {'train': loss_training,
                                         'val': loss_validation
                                         }, epoch)

                self.scheduler.step(loss_validation)

                # early stopping
                self.early_stopping(loss_validation, self.model, epoch)

                if self.early_stopping.early_stop:
                    print('Early stopping')
                    break

        except KeyboardInterrupt:
            print('Stopped training through KeyboardInterrupt')

        # logging
        time_elapsed = time.time() - start_time

        print(f'Training finished in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.')
        print('Last epoch results:')
        for key, value in self.epochs_stats[-1].items():
            print(f'\t{key}: {value:.4f}')

        # load best model checkpoint
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        print(f'Loading checkpoint at epoch {self.early_stopping.best_epoch}\n'
              f'Checkpoint results:')

        # best epoch results (checkpoint)
        self.best_results = self.epochs_stats[self.early_stopping.best_epoch]
        for key, value in self.best_results.items():
            print(f'\t{key}: {value:.4f}')

        # logging
        wb.log(self.best_results)

        self.epochs_trained += epochs

        # evaluate on test set
        self.test_results = self.eval_test()
        wb.log(self.test_results)

        # evaluate on real test data
        real_test_results = self.eval_real_test()
        wb.log(real_test_results)

        self.writer.flush()
        self.writer.close()

    def eval_test(self):
        """
        Test the model on the test set.

        Samples taken one by one
        """
        gts = []
        predictions = []
        losses = []

        self.model.eval()
        print('Evaluating model on test subset:')

        with torch.no_grad():
            with tqdm.tqdm(enumerate(self.test_dataset)) as pbar:
                for i, sample in pbar:
                    x, y = sample
                    gts.append(y)

                    x = x.to(device=self.device, dtype=torch.float)
                    y = y.to(device=self.device, dtype=torch.float)
                    pred = self.model.predict(x)[:, 0]
                    loss = self.bce_loss(pred, y)

                    # logging
                    loss_value = loss.item()
                    losses.append(loss_value)
                    predictions.append(pred)

                    pbar.set_postfix(loss=f'{loss_value:.4f}')

        loss = np.nanmean(losses)
        accu = accuracy(gts, predictions)['accuracy']
        print(f'Test set results:\n'
              f'Loss: {loss:.4f}\n'
              f'Accuracy: {accu:.4f}'
              )
        return {'Loss Test': loss,
                'Accuracy Test': accu}

    def save_model(self, path):
        """
        Save model to path
        """
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to "{path}"')

    def print_setup(self):
        print(self.model)
        count_parameters(self.model)
        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')


def balance(dataset):
    ys = []
    for s in tqdm.tqdm(dataset):
        """
        real test set:
        ValueError: too many values to unpack (expected 2)
        """

        x, y = s
        ys.append(y)
    ys = torch.cat(ys)
    return float(torch.mean(ys))


# def main():
if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Running on device: {device}')

    args_global = parser.parse_args()
    print('Args:\n', get_dict(args_global))

    config.dataset_preprocessed_path = args_global.source

    # con = 'shared_resources/real_test_for_milestone3/real_consistent.json'
    # incon = 'shared_resources/real_test_for_milestone3/real_inconsistent.json'
    # dataset_path = 'shared_resources/dataset_preprocessed_1000.pkl'

    # config.dataset_preprocessed_path = args.source

    model_training = ModelTraining()
    #
    model_training.train_model(epochs=config.HPARS['epochs'])
    model_training.save_model(args_global.destination)
    #
    self = model_training
    s1 = next(iter(model_training.train_dataset))
    s2 = next(iter(model_training.val_dataset))
    s3 = next(iter(model_training.test_dataset))

    p1 = model_training.pred_sample(s1)
    p2 = model_training.pred_sample(s2)
    p3 = model_training.pred_sample(s3)

    """
    
    # dataset stats
    balance_train = balance(model_training.train_dataset)
    balance_val = balance(model_training.val_dataset)
    balance_test = balance(model_training.test_dataset)
    print(f'Balance train: {balance_train:.4f}\n'
          f'Balance val:   {balance_val:.4f}\n'
          f'Balance test:  {balance_test:.4f}\n')
    wb.log({'balance_train': balance_train, 'balance_val': balance_val, 'balance_test': balance_test})


    plt.plot(model_training.training_losses); plt.show()

    print('Training Dataset Balance', )
    """

# if __name__ == '__main__':
#     main()
