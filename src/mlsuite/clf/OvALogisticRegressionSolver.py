import numpy as np

from mlsuite.protocol import FloatArrayT
from .Conf import LogisticClfConfig


class OvALogisticRegressionSolver:
    def __init__(self, conf: LogisticClfConfig):
        self.hp = conf

    def fit(self, X_train: FloatArrayT, y_train: FloatArrayT):
        # ! Assumes class numbers start from 0 and grow discretely.
        if np.max(y_train) >= self.hp.num_classes or np.min(y_train) < 0:
            raise ValueError(
                f"OvALogisticRegressionSolver was initialized with a different parameter: self.hp.num_classes={self.hp.num_classes}"
            )
        # Since there are weights for each class, we use
        # W = (D, C), where
        #   D — number of features
        #   C - number of classes
        # X = (N, D)
        #   N — number of data points
        # Each class has its own weights, we
        # apply each to data point and use.
        N, D = X_train.shape

        # Train a separate binary classification model for each class.
        # Put them all into a single matrix for efficiency.
        W = np.random.randn(D, self.hp.num_classes) * 0.01
        if self.hp.use_bias:
            B = np.zeros((1, self.hp.num_classes))
        else:
            B = 0

        def _compute_hypothesis():
            nonlocal X_train, W, B
            """
            Computes the value of the hypothesis according to the logistic regression rule.
            Y = sigmoid(W*X + b)
            """
            Z = X_train @ W + B
            return self._calc_matrix_sigmoid(Z)

        Y = np.zeros((N, self.hp.num_classes))
        Y[np.arange(N), y_train] = 1

        lr = self.hp.lr
        for _ in range(self.hp.niters):
            # Compute hypothesis.
            A = _compute_hypothesis()
            # Calculate error.
            pure_error = A - Y  # (N, C)
            # For Cross-Entropy, sigmoid function cancels its own derivative.
            # Update the weights and biases accordingly.
            W = W - lr * (
                (1 / N) * X_train.T @ pure_error  # Gradient (D, C)
                + (self.hp.l2_coef * W)  # L2 Regularization
            )
            if self.hp.use_bias:
                B = B - lr * (1 / N) * np.sum(pure_error, axis=0, keepdims=True)
            if self.hp.diminishing_lr:
                lr *= self.hp.lr_dim_coef
        return W, B

    def _calc_matrix_sigmoid(self, Z: FloatArrayT) -> FloatArrayT:
        """Returns an array of the same shape as Z, where each element is the sigmoid of the corresponding element in Z."""
        sigmoid_val = 1 / (1 + np.exp(-Z))
        return sigmoid_val
