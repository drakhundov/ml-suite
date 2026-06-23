import numpy as np

from mlsuite.protocol import FloatArrayT
from .Conf import LogisticClfConfig

class OvALogisticRegressionSolver:
    def __init__(self, conf: LogisticClfConfig):
        self.hp = conf

    def fit(self, X_train: FloatArrayT, y_train: FloatArrayT):
        # ! Assumes class numbers start from 0 and discretely go up.
        if np.max(y_train) > self.hp.num_classes:
            raise ValueError(f"OvALogisticRegressionSolver was initialized with a different parameter: self.hp.num_classes={self.hp.num_classes}")
        # Since there are weights for each class, we use
        # W = (C, D), where
        #   C - number of classes
        #   D — number of features
        # X = (D, N)
        #   N — number of data points
        X_train = X_train.T
        D, N = X_train.shape

        # Train a separate binary classification model for each class.
        # Put them all into a single matrix for efficiency.
        W = np.random.randn(self.hp.num_classes, D) * 0.01
        if self.hp.use_bias:
            B = np.zeros((self.hp.num_classes, 1))
        else:
            B = None

        def _compute_hypothesis():
            nonlocal X_train, W, B
            """
            Computes the value of the hypothesis according to the logistic regression rule.
            Y = sigmoid(W*X + b)
            """
            Z = W @ X_train + B
            h_theta = self._calc_matrix_sigmoid(Z)
            return h_theta

        Y = np.zeros((self.hp.num_classes, N))
        Y[y_train, np.arange(N)] = 1

        lr = self.hp.lr
        for _ in range(self.hp.niters):
            # Compute hypothesis.
            A = _compute_hypothesis()
            # Calculate error.
            pure_error = A - Y
            # For Cross-Entropy, sigmoid function cancels its own derivative.
            # Update the weights and biases accordingly.
            if self.hp.diminishing_lr:
                lr *= self.hp.lr_dim_coef
            W = W - lr * (
                (1 / N) * pure_error @ X_train.T  # Gradient
                + (self.hp.l2_coef * W)      # L2 Regularization
            )
            if self.hp.use_bias:
                B = B - lr * (1 / N) * np.sum(
                    pure_error, axis=1, keepdims=True
                )
        return W, B

    def _calc_matrix_sigmoid(self, Z: FloatArrayT) -> FloatArrayT:
        """Returns an array of the same shape as Z, where each element is the sigmoid of the corresponding element in Z."""
        sigmoid_val = 1 / (1 + np.exp(-Z))
        return sigmoid_val
