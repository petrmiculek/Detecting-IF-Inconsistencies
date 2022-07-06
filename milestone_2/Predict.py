#!/usr/bin/python3
import json
import os
import sys
import shlex

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import torch.nn as nn



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


def predict(model, test_files):
    """
    Predict inconsistencies.
    """
    test_files = ["shared_resources/real_test_for_milestone3/real_consistent.json"]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = load_tokenizer(config.model_input_len)
    bce_loss = nn.BCELoss()

    all_outputs = []
    for f in test_files:
        predictions = []
        gt = []
        lines = []
        dataset = IfRaisesDataset(f, tokenizer, fraction=1., eval_mode=True)
        # dataset, _, _ = get_dataset_loaders(f, tokenizer, training_split=1., batch_size=1, eval_mode=True)

        mean_loss = 0
        for i, s in tqdm.tqdm(enumerate(dataset)):
            with torch.no_grad():
                x, y, line = s
                x = torch.tensor(x).to(device)
                y = torch.tensor(y).to(device)
                pred = model(x)
                loss = bce_loss(pred[0], y)
                mean_loss += loss.item()
                predictions.append(float(pred[0].item()))
                gt.append(y.item())
                lines.append(int(line))

        mean_loss /= len(dataset)

        output = dict(sorted(zip(lines, predictions)))
        all_outputs.append(output)
        correct = np.array(predictions) == np.array(gt)

        print(f'Mean loss: {mean_loss}')
        print(f'Accuracy: {correct.sum() / len(correct)}')

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

    source = "shared_resources/model_weights.pt"
    model = LSTM(config.model_input_len, config.model['hidden_size'])
    model.load_state_dict(torch.load(source))
    """
    RuntimeError: Error(s) in loading state_dict for LSTMBase:
    size mismatch for lstm.weight_ih_l0: copying a param with shape torch.Size([512, 32]) 
    from checkpoint, the shape in current model is torch.Size([512, 256]).
    """
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

    # load the serialized model
    # model = load_model(args.model)

    # predict incorrect location for each test example.
    predictions = predict(lambda x: torch.tensor([0.0], device=device), args.source)

    # write predictions to file
    write_predictions(args.destination, predictions)
