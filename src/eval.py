from os.path import join
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix as conf_mat, classification_report
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, RocCurveDisplay, auc


def compute_metrics(gts, predictions):
    # todo choose where to concat list into tensor
    gts = torch.cat(gts)
    predictions = torch.cat(predictions)

    # todo unified call for evaluation
    predictions_hard = torch.tensor([1 if x >= 0.5 else 0 for x in predictions])

    print(classification_report(gts, predictions_hard))

    # unused
    # confusion_matrix(gts, predictions_hard)

    results = {}
    for metric in [accuracy]:
        results.update(metric(gts, predictions, predictions_hard))
    return results


def accuracy(gts, predictions, predictions_hard=None):
    """
    Compute metrics from predictions and ground truth
    :param gts: ground truth
    :param predictions: predictions
    :param predictions_hard: prediction decisions
    """
    if type(gts) == list:
        gts = torch.cat(gts).cpu()
        predictions = torch.cat(predictions).cpu()

    if predictions_hard is None:
        # predictions_hard = torch.tensor([1 if x >= 0.5 else 0 for x in predictions])
        predictions_hard = (predictions >= 0.5)

    correct = predictions_hard == gts

    accuracy = torch.sum(correct) / len(correct)

    results = {
        'accuracy': accuracy.item()
    }
    return results


def confusion_matrix(gts, predictions_hard, output_location=None, show=True, val=False, normalize=True):
    """Create and show/save confusion matrix"""
    label_names = ['consistent', 'inconsistent']
    model_name = 'LSTM_Base'
    epochs_trained = '20'
    kwargs = {}
    if normalize:
        # {'true', 'pred', 'all'}, default = None
        normalize = 'true'
        kwargs['vmin'] = 0.0
        kwargs['vmax'] = 1.0
        kwargs['fmt'] = '0.2f'
    else:
        normalize = None
        kwargs['fmt'] = 'd'

    cm = conf_mat(list(gts), list(predictions_hard), normalize=normalize)

    sns.set_context('paper', font_scale=1.8)
    fig_cm = sns.heatmap(
        cm,
        annot=True,
        xticklabels=label_names,
        yticklabels=label_names,
        # fmt='0.2f',
        # vmin=0.0,
        # vmax=1.0
        **kwargs
    )
    fig_cm.set_title('Confusion Matrix\n{} {} [e{}]'
                     .format(model_name, 'val' if val else 'train', epochs_trained))
    fig_cm.set_xlabel('Predicted')
    fig_cm.set_ylabel('True')
    fig_cm.axis('on')
    fig_cm.figure.tight_layout(pad=0.5)

    if show:
        fig_cm.figure.show()

    if output_location:
        fig_cm.figure.savefig(join(output_location, 'confusion_matrix' + '_val' * val + '.svg'),
                              bbox_inches='tight')

    plt.close(fig_cm.figure)


def plot_roc_curve(gts, predictions, _, show=True, output_location=None):

    fpr, tpr, thresholds = roc_curve(gts, predictions)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
    display.plot()
    if show:
        plt.show()
    if output_location is not None:
        plt.savefig(join(output_location, 'roc_curve.svg'), bbox_inches='tight')
