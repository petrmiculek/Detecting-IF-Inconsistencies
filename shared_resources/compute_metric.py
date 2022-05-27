# Compute com
import json
import os
import argparse
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument(
        '--reference', help="Path of reference files.", required=True)
parser.add_argument(
        '--prediction', help="Path to output json file of extracted predictions.", required=True)


def read_file(file_name):
    data_obj = None
    with open(file_name) as f:
        data_obj= json.load(f)
    return data_obj
    


if __name__ == "__main__":

    args = parser.parse_args()
    
    # read the reference input
    references = read_file(args.reference)

    # read predicted file
    predictions = read_file(args.prediction)
    
    ref_labels = []
    predicted_labels = []
    

    for idx,reference in enumerate(references):
        if idx <= len(predictions):
            prediction = predictions[idx]
            for if_cond_idx, ref_obj in enumerate(reference):
                if if_cond_idx <= len(prediction):
                    predicted_labels.append(prediction[if_cond_idx]["consistency"])
                    ref_labels.append(ref_obj["consistency"])
    #
    precision =  precision_score(ref_labels,predicted_labels)
    recall =  recall_score(ref_labels,predicted_labels)
    f1 =  f1_score(ref_labels,predicted_labels)
    accuracy_score =  accuracy_score(ref_labels,predicted_labels)
    print(f"Precision: {precision},Recall: {recall},F1_score: {f1},accuracy: {accuracy_score}")