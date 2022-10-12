import numpy as np
import torch.nn as nn
import torch.nn.init as init
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, matthews_corrcoef

# Early Stopping
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=100, delta=1e-3):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 100
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        elif score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.early_stop = False

        else:
            self.best_score = score
            self.counter = 0

# Binary Class
def calculate_metric(y_true, y_pred, y_pred_proba):
    y_pred_proba = y_pred_proba[:, 1]
    ba = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    mcc = matthews_corrcoef(y_true, y_pred)

    return ba, f1, auc, mcc

# Weight & Bias Initialization
def initialization(net):
    if isinstance(net, nn.Linear):
        init.kaiming_uniform_(net.weight)
        init.zeros_(net.bias)