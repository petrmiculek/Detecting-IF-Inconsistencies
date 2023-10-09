# Detecting IF-Inconsistencies
tl;dr I used an LSTM to classify if-condition-statement code blocks as consistent/inconsistent (think erroneous).

Using a provided codebase, I parsed the code into an abstract syntax tree (AST), extracted if-condition-statements, and transformed them into a dataset. Creating inconsistent but plausible synthetic data for training was an interesting problem to tackle here (see slides for more). I trained a neural network (bi-directional single-layer LSTM with a classifier head) to classify the statements as consistent or inconsistent (containing a bug or a typo). The solution reached an 87.4% accuracy in the synthetic dataset, but the performance did not generalize to real test cases (outside the dataset).
In this project, I used: Python, Pytorch, Huggingface, Numpy, Pandas, LibCST, Scikit-Learn, Weights&Biases, a FastText tokenizer, and maybe a few other toys.
This project was developed in the "Analysing Software with Deep Learning" course at Universität Stuttgart as a solo project under the supervision of Islem Bouzenia.

## Presentation
[Google Slides link](https://docs.google.com/presentation/d/1JmDiugDL0A8nDBmi5EovO7dNUadN7gxwbh0FiwOqMNE/edit?usp=sharing)

## How to run
* Create and activate a Python3.8 env
* `pip install -r requirements.txt`
* download shared resources (as per assignment instructions)
* `python3 milestone2/Train.py --source <dataset source dir> --destination <output trained model path>`
* `python3 milestone2/Predict.py --model <model path> --source <testing inputs> --destination <output JSON predictions>`

## Dataset information
Ifs: 357866
Elses: 59717
Elifs: 34858
Raises: 155274
