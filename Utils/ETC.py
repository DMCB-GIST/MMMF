import os
import math
import torch
import random
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.init as init
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, matthews_corrcoef

# Ignore Warning Massage => All class 0 or 1
warnings.filterwarnings(action='ignore')

# Calculate Metric
def calculate_metric(y_true, y_pred, y_pred_proba, multi_class=False):
    ba = balanced_accuracy_score(y_true, y_pred)

    # Binary Class
    if not multi_class:
        y_pred_proba = y_pred_proba[:, 1]
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_proba)
        mcc = matthews_corrcoef(y_true, y_pred)

    # Multi-Class => BRCA
    else:
        f1 = f1_score(y_true, y_pred, average='weighted')
        auc = roc_auc_score(y_true, y_pred_proba, average='weighted', multi_class='ovo')
        mcc = matthews_corrcoef(y_true, y_pred)

    return ba, f1, auc, mcc

# Set Seed
def set_seed():
    # Seed Setting
    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    torch.cuda.manual_seed(100)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_num_threads(10)

# Weight & Bias Initialization
def initialization(net):
    if isinstance(net, nn.Linear):
        init.kaiming_uniform_(net.weight)
        init.zeros_(net.bias)

# Create Directory
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def to_precision(x,p):
    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)

    if n < math.pow(10, p - 1):
        e = e -1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= math.pow(10,p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('E')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)

    return "".join(out)

# Save Result
def Result_Save(result, save_path, all_hyper_parameter, one_hyper):
    # Save Path
    val_save_path = os.path.join(os.path.join(save_path), 'Validation.csv')
    test_save_path = os.path.join(os.path.join(save_path), 'Test.csv')

    col = ['BA', 'F1', 'AUC', 'MCC']
    index = all_hyper_parameter

    if not os.path.isfile(val_save_path):
        result_df = pd.DataFrame(columns=col, index=index)
        result_df.to_csv(val_save_path)
        result_df.to_csv(test_save_path)

    # Save Result
    val_result_df = pd.read_csv(val_save_path, index_col=0)
    test_result_df = pd.read_csv(test_save_path, index_col=0)

    # Save Validation Result
    for c in col:
        val_result_df.loc[one_hyper, c] = ', '.join(['{:.4f}'.format(x) for x in result['Validation'][c]])
    val_result_df.to_csv(val_save_path)

    # Save Test Result
    for c in col:
        test_result_df.loc[one_hyper, c] = ', '.join(['{:.4f}'.format(x) for x in result['Test'][c]])
    test_result_df.to_csv(test_save_path)

def Random_Result_Save(result, save_path, all_hyper_parameter, one_hyper, random_state):
    # Save Path
    val_save_path = os.path.join(os.path.join(save_path), 'Validation'+str(random_state)+'.csv')
    test_save_path = os.path.join(os.path.join(save_path), 'Test'+str(random_state)+'.csv')

    col = ['BA', 'F1', 'AUC', 'MCC']
    index = all_hyper_parameter

    if not os.path.isfile(val_save_path):
        result_df = pd.DataFrame(columns=col, index=index)
        result_df.to_csv(val_save_path)
        result_df.to_csv(test_save_path)

    # Save Result
    val_result_df = pd.read_csv(val_save_path, index_col=0)
    test_result_df = pd.read_csv(test_save_path, index_col=0)

    # Save Validation Result
    for c in col:
        val_result_df.loc[one_hyper, c] = ', '.join(['{:.4f}'.format(x) for x in result['Validation'][c]])
    val_result_df.to_csv(val_save_path)

    # Save Test Result
    for c in col:
        test_result_df.loc[one_hyper, c] = ', '.join(['{:.4f}'.format(x) for x in result['Test'][c]])
    test_result_df.to_csv(test_save_path)

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