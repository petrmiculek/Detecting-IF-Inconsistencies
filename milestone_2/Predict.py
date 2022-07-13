#!/usr/bin/python3
import json
import os
import sys
import shlex

import sys

sys.path.extend(['/home/petrmiculek/Code/asdl'])

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import torch.nn as nn
from sklearn.metrics import roc_curve

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', help="Path of trained model.", required=True)
parser.add_argument(
    '--source', help="Path of the test function file.", required=True)
parser.add_argument(
    '--destination', help="Path to output JSON file with predictions.", required=True)

from src import config
from src.extract import load_segments, extract_raises
from src.preprocess import load_tokenizer
from src.dataset import IfRaisesDataset, get_dataset_loaders
from src.model import LSTMBase as LSTM
from src.eval import compute_metrics, accuracy


def predict(model, test_files):
    """
    Predict inconsistencies.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = load_tokenizer(config.model_input_len)

    all_outputs = []

    # for test_file in test_files:  # apparently no
    predictions = []
    lines = []
    print('Extracting test data...')
    dataset = IfRaisesDataset(test_files, tokenizer, fraction=1., eval_mode=True)

    print('Running predictions...')
    for i, s in tqdm.tqdm(enumerate(dataset)):
        with torch.no_grad():
            x, _, line = s
            x = torch.tensor(x).to(device)
            pred = model(x)
            predictions.append(float(pred[0].item()))
            lines.append(int(line))

    predictions_hard = [1 if x >= 0.5 else 0 for x in predictions]
    output = dict(sorted(zip(lines, predictions_hard)))
    all_outputs.append(output)

    return all_outputs


def evaluate(model, test_files):
    """
    Predict inconsistencies.
    """
    # fake_model = lambda x: torch.tensor([0.0], device=device)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = load_tokenizer(config.model_input_len)
    bce_loss = nn.BCELoss()

    all_outputs = []

    # for test_file in test_files:  # apparently no
    predictions = []
    gts = []
    lines = []
    print('Extracting test data...')
    dataset = IfRaisesDataset(test_files, tokenizer, fraction=1., eval_mode=True)
    # dataset, _, _ = get_dataset_loaders(f, tokenizer, training_split=1., batch_size=1, eval_mode=True)

    print('Running predictions...')
    mean_loss = 0
    for i, s in tqdm.tqdm(enumerate(dataset)):
        with torch.no_grad():
            x, y, line = s
            y += 1

            gts.append(y)
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)
            pred = model(x)
            loss = bce_loss(pred[0], y)
            mean_loss += loss.item()
            predictions.append(float(pred[0].item()))
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

    model = LSTM(config.model['input_size'], config.model['hidden_size'])
    model.load_state_dict(torch.load(source))
    model.to(device)
    model.eval()

    return model


# def load_bert(source):
#
#     CODEBERTA_LANGUAGE_ID = "huggingface/CodeBERTa-language-id"
#
#     from transformers import RobertaTokenizer
#     from transformers import RobertaForSequenceClassification
#     tokenizer = RobertaTokenizer.from_pretrained(CODEBERTA_LANGUAGE_ID)
#     model = RobertaForSequenceClassification.from_pretrained(CODEBERTA_LANGUAGE_ID)
#
#     input_ids = tokenizer.encode(CODE_TO_IDENTIFY)
#     logits = model(input_ids)[0]
#
#     language_idx = logits.argmax()
#     return language_idx
#     # todo try

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

    # load the serialized model
    model = load_model(args.model)
    # predict incorrect location for each test example.
    predictions = predict(model, args.source)
    # write predictions to file
    write_predictions(args.destination, predictions)
    # print(predictions)
