#!/usr/bin/python3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', help="Path to functions file used for training.", required=True)
parser.add_argument(
    '--destination', help="Path to save your trained model.", required=True)


def train_model(model, source):
    """
    TO DO: Implement your method for training the model here.
    """
    raise Exception("Method not yet implemented.")


def save_model(model, destination):
    """
    TO DO: Implement your method for saving the training the model here.
    """
    raise Exception("Method not yet implemented.")


if __name__ == "__main__":
    args = parser.parse_args()

    model = train_model(args.source)

    save_model(model, args.destination)

    # continuing my work in milestone_2
