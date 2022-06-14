#!/usr/bin/python3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', help="Path of trained model.", required=True)
parser.add_argument(
    '--source', help="Path of the test function file.", required=True)
parser.add_argument(
    '--destination', help="Path to output JSON file with predictions.", required=True)


def predict(model, test_files):
    """
    TODO: Implement your method for predicting inconsistencies.
    """
    raise Exception("Method not yet implemented.")


def load_model(source):
    """
    TODO: Implement your code to load the serialized model.
    """
    # raise Exception("Method not yet implemented.")
    return None


def write_predictions(destination, predictions):
    """
    TODO: Implement your code to write predictions to file. For format
    of the JSON file refer to project description.
    """
    raise Exception("Method not yet implemented.")


if __name__ == "__main__":
    args = parser.parse_args()

    # load the serialized model
    model = load_model(args.model)

    # predict incorrect location for each test example.
    predictions = predict(model, args.source)

    # write predictions to file
    write_predictions(args.destination, predictions)
