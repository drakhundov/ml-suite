from dataclasses import dataclass
from typing import Union, List

import numpy as np


@dataclass(frozen=True)
class TestStats:
    TP: int
    TN: int
    FP: int
    FN: int
    accuracy: float
    recall: float
    precision: float
    F1: float
    MSE: float


def produce_test_stats_reg(preds: Union[List, np.ndarray], y_true, eps: float = 1e-9):
    preds = np.asarray(preds).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    if preds.ndim == 2:
        if preds.shape[0] != y_true.shape[0]:
            raise ValueError(
                "The number of predictions does not match the number of true labels"
            )
    elif preds.ndim > 2:
        raise ValueError(
            "The number of predictions does not match the number of true labels"
        )
    if y_true.ndim != 1:
        raise ValueError(
            "The number of true labels does not match the number of predictions"
        )
    acc = np.sum((preds - y_true) <= eps) / len(y_true)
    mse = np.sum((preds - y_true) ** 2) / len(y_true)
    return TestStats(
        TP=0, TN=0, FP=0, FN=0, accuracy=acc, recall=0.0, precision=0.0, F1=0.0, MSE=mse
    )


def produce_test_stats_clf(preds: Union[List, np.ndarray], y_true, _class: int = 0):
    preds = np.asarray(preds).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    if preds.ndim == 2:
        if preds.shape[0] != y_true.shape[0]:
            raise ValueError(
                "The number of predictions does not match the number of true labels"
            )
    elif preds.ndim > 2:
        raise ValueError(
            "The number of predictions does not match the number of true labels"
        )
    if y_true.ndim != 1:
        raise ValueError(
            "The number of true labels does not match the number of predictions"
        )
    tp = np.sum((preds == _class) & (y_true == _class))
    tn = np.sum((preds != _class) & (y_true != _class))
    fp = np.sum((preds == _class) & (y_true != _class))
    fn = np.sum((preds != _class) & (y_true == _class))
    if (div := (tp + tn + fp + fn)) == 0:
        acc = 0
    else:
        acc = (tp + tn) / div
    if (div := (tp + fn)) == 0:
        rec = 0
    else:
        rec = tp / div
    if (div := (tp + fp)) == 0:
        prec = 0
    else:
        prec = tp / div
    if (div := (tp + fp + fn)) == 0:
        f1 = 0
    else:
        f1 = 2 * tp / div
    return TestStats(
        TP=tp,
        TN=tn,
        FP=fp,
        FN=fn,
        accuracy=acc,
        recall=rec,
        precision=prec,
        F1=f1,
        MSE=0.0,
    )
