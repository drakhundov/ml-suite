from dataclasses import dataclass
from typing import Union, List

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from clf import KNNClassifier, KMCClassifier


@dataclass(frozen=True)
class LearnStats:
    TP: int
    TN: int
    FP: int
    FN: int
    accuracy: int
    recall: int
    precision: int
    F1: int


def produce_pred_stats(preds: Union[List, np.ndarray], y_true, _class: int = 0):
    preds = np.asarray(preds)
    y_true = np.asarray(y_true)
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
    return LearnStats(
        TP=tp,
        TN=tn,
        FP=fp,
        FN=fn,
        accuracy=(tp + tn) / (tp + tn + fp + fn),
        recall=tp / (tp + fn),
        precision=tp / (tp + fp),
        F1=2 * tp / (tp + fp + fn),
    )


class IrisLearner:
    def __init__(
        self,
        clf: Union[KNNClassifier, KMCClassifier],
        test_sz: float = 0.2,
        split_by_sk: bool = False,
    ):
        self.clf = clf
        self.split_by_sk = split_by_sk
        iris = load_iris()
        X, y = iris.data, iris.target
        self.dataset_sz = X.shape[0]
        self.n_train = (1 - test_sz) * self.dataset_sz
        if self.split_by_sk:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_sz, random_state=42, stratify=y
            )
        else:
            self.X_train, self.X_test = X[: self.n_train], X[self.n_train :]
            self.y_train, self.y_test = y[: self.n_train], y[self.n_train :]

    def fit_and_get_stats(self) -> List[LearnStats]:
        if isinstance(self.clf, KNNClassifier):
            data = (self.X_train, self.y_train,)
        else:
            data = (self.X_train,)
        self.clf.fit(*data)
        test_pred = np.array(self.clf.predict(self.X_test))
        stats = []
        for i in range(3):
            # Iris dataset contains three classes.
            stats.append(produce_pred_stats(test_pred, self.y_test, _class=i))
        return stats
