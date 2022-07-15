#!/usr/bin/python3
import json
import os
import shlex
import sys

sys.path.extend(['/home/petrmiculek/Code/asdl'])

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import roc_curve

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', help="Path of trained model.", required=True)
parser.add_argument(
    '--source', help="Path of the test function file.", required=True)
parser.add_argument(
    '--destination', help="Path to output JSON file with predictions.", required=True)

from src import config
from src.dataset import IfRaisesDataset, get_dataset_loaders
from src.eval import accuracy, compute_metrics
from src.extract import extract_raises, load_segments
from src.model import LSTMBase as LSTM
from src.preprocess import load_tokenizer


def predict(model, test_files):
    """
    Predict inconsistencies.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = load_tokenizer(config.tokens_length_unit)

    all_outputs = []

    # for test_file in test_files:  # apparently no
    predictions = []
    lines = []
    print('Extracting test data...')
    dataset, _, _ = get_dataset_loaders(test_files, tokenizer, training_split=1., batch_size=2, eval_mode=True)

    print('Running predictions...')
    with torch.no_grad():
        for s in tqdm.tqdm(dataset):
            x, _, line = s
            x = x.to(device=device)
            pred = model.predict(x)[:, 0]
            predictions.append(pred)
            lines.append(line)

    predictions = torch.cat(predictions)
    predictions_hard = [1 if x >= 0.5 else 0 for x in predictions]
    lines = list(torch.cat(lines).cpu().numpy().astype(int))
    output = dict(sorted(zip(lines, predictions_hard)))
    all_outputs.append(output)

    return all_outputs


def evaluate(model, test_files):
    """
    Predict inconsistencies.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = load_tokenizer(config.tokens_length_unit)
    bce_loss = nn.BCEWithLogitsLoss()

    all_outputs = []

    # for test_file in test_files:  # apparently no
    predictions = []
    gts = []
    lines = []
    print('Extracting test data...')
    # dataset = IfRaisesDataset(test_files, tokenizer, fraction=1., eval_mode=True)
    dataset, _, _ = get_dataset_loaders(test_files, tokenizer, training_split=1., batch_size=1, eval_mode=True)

    print('Running predictions...')
    mean_loss = 0
    for i, s in tqdm.tqdm(dataset):
        with torch.no_grad():
            x, y, line = s
            y += 1

            gts.append(y)
            x = x.to(device=device)
            y = y.to(device=device)
            pred = model(x)[:, 0]
            loss = bce_loss(pred, y)
            mean_loss += loss.item()
            predictions.append(float(pred.item()))
            lines.append(int(line))

    mean_loss /= len(dataset)

    output = dict(sorted(zip(lines, predictions)))
    all_outputs.append(output)

    # todo test
    compute_metrics(gts, predictions)

    print(f'Mean loss: {mean_loss:.4f}')

    return all_outputs


def model_predict(x):
    """
    interface dummy - from dataframe to target pred format
    """
    result_dict = np.zeros(len(x))
    result_dict = [{k: v for (k, v) in zip(x['line_raise'], result_dict)}]
    return result_dict


def load_model(source):
    """
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = LSTM(config.model['input_size'], config.model['hidden_size'], tokens_length=config.model_input_len)
    # model.load_state_dict(torch.load(source))
    model.to(device)
    model.eval()

    return model

def write_predictions(destination, predictions):
    """
    Write predictions to file
    in the JSON format as per project description.
    """
    with open(destination, 'w') as f:
        f.write(json.dumps(predictions))


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # args = parser.parse_args(shlex.split("--model shared_resources/model_weights.pt "
    #                          "--source shared_resources/real_test_for_milestone3/real_consistent.json "
    #                          "--destination shared_resources/predictions.json"))
    args = parser.parse_args()
    test_files = args.source

    # load the serialized model
    model = load_model(args.model)
    # predict incorrect location for each test example.
    predictions_formatted = predict(model, test_files)
    # write predictions to file
    write_predictions(args.destination, predictions_formatted)
    # print(predictions)


