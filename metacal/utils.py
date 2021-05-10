from pathlib import Path
import pickle
import socket

import numpy as np
from sklearn.model_selection import train_test_split

from metacal.expconf import logit_names

logit_base = Path("")


def load_data(conf_name, repeat=False):
    """
    repeat: if True, first merge train and test, then do a random split
    """
    assert conf_name in logit_names

    def prepare_data(name):
        with open(name, "rb") as f:
            (logits_train, y_train), (logits_test, y_test) = pickle.load(f)
        return logits_train, y_train.flatten(), logits_test, y_test.flatten()

    logit_name = f"probs_{conf_name}_logits.p"
    logit_train, y_train, logit_test, y_test = prepare_data(logit_base / logit_name)

    if repeat:
        train_size = logit_train.shape[0]
        _logit_all = np.r_[logit_train, logit_test]
        _y_all = np.r_[y_train, y_test]
        logit_train, logit_test, y_train, y_test = train_test_split(
            _logit_all, _y_all, train_size=train_size
        )

    return logit_train, y_train, logit_test, y_test


def errors(preds, targets):
    """ compute type1 error, type2 error, and accuracy

    preds: (N,)
    targets: (N,)
    """
    preds = np.asarray(preds).astype("i")
    targets = np.asarray(targets).astype("i")
    indices = targets == 0
    R0 = np.mean(preds[indices] != targets[indices])  # type-I error
    R1 = np.mean(preds[~indices] != targets[~indices])  # type-II error
    acc = np.mean(preds == targets)
    w1 = np.mean(targets)
    w0 = 1 - w1
    return R0, R1, w0, w1, acc
