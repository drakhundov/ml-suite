from typing import Optional

import numpy as np

from mlsuite.protocol import FloatArrayT
from .Conf import LogisticClfConfig
from .OvALogisticRegressionSolver import OvALogisticRegressionSolver


class OvALogisticClassification:
    def __init__(self, conf: LogisticClfConfig):
        # Model hyperparameters.
        self.hp = conf
        self.solver = OvALogisticRegressionSolver(conf)

    def fit(self, X_train: FloatArrayT, y_train: FloatArrayT, dtype: Optional[np.dtype] = None):
        self.X_train = X_train
        self.y_train = y_train

        # Train a separate binary classification model for each class.
        # Put them all into a single matrix for efficiency.
        self.W, self.B = self.solver.fit(X_train, y_train, dtype)

    def predict(self, X_new: FloatArrayT) -> FloatArrayT:
        # X_new -> (N, D)
        #   N — number of data points
        #   D — number of features
        return self.solver._compute_hypothesis(X_new)

    def get_weights_for_class(self, classno: int) -> FloatArrayT:
        w = self._W[classno, :]
        return w
