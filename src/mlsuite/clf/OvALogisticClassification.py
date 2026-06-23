import numpy as np

from mlsuite.protocol import FloatArrayT
from .Conf import LogisticClfConfig
from .OvALogisticRegressionSolver import OvALogisticRegressionSolver


class OvALogisticClassification:
    def __init__(
        self,
        conf: LogisticClfConfig
    ):
        # Model hyperparameters.
        self.hp = conf
        self.solver = OvALogisticRegressionSolver(conf)

    def fit(self, X_train: FloatArrayT, y_train: FloatArrayT):
        self.X_train = X_train
        self.y_train = y_train

        # Train a separate binary classification model for each class.
        # Put them all into a single matrix for efficiency.
        self.W, self.B = self.solver.fit(X_train, y_train)

    def predict(self, X_new: FloatArrayT) -> FloatArrayT:
        H = self._calc_matrix_sigmoid(self.W.dot(X_new) + self.B)
        return H

    def _calc_matrix_sigmoid(self, Z: FloatArrayT) -> FloatArrayT:
        """Returns an array of the same shape as Z, where each element is the sigmoid of the corresponding element in Z."""
        sigmoid_val = 1 / (1 + np.exp(-Z))
        return sigmoid_val

    def get_weights_for_class(self, classno: int) -> FloatArrayT:
        w = self._W[classno, :]
        return w
