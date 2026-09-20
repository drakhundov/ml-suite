from typing import Optional

import numpy as np
from IPython.display import HTML, display

from mlsuite.protocol import FloatArrayT
from .Conf import LogisticClfConfig


class OvALogisticRegressionSolver:
    def __init__(self, conf: LogisticClfConfig):
        self.hp = conf

    def fit(self, X_train: FloatArrayT, y_train: FloatArrayT, dtype: Optional[np.dtype] = None):
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
        if N != y_train.size:
            raise ValueError(
                f"X_train(shape {X_train.shape}) and y_train(shape {y_train.shape}) don't match"
            )
        
        if dtype is None:
            dtype = X_train.dtype
        else:
            X_train = X_train.astype(dtype)

        # Train a separate binary classification model for each class.
        # Put them all into a single matrix for efficiency.
        self.W = np.random.randn(D, self.hp.num_classes).astype(dtype) * 0.01
        if self.hp.use_bias:
            self.B = np.zeros((1, self.hp.num_classes)).astype(dtype)
        else:
            self.B = 0

        Y = np.zeros((N, self.hp.num_classes))
        Y[np.arange(N), y_train] = 1

        lr = self.hp.lr
        progress_bar = display(
            HTML(
                '<progress value="0" max="{}"></progress> 0%'.format(
                    self.hp.niters
                )
            ),
            display_id=True,
        )
        for _ in range(self.hp.niters):
            # Compute hypothesis.
            A = self._compute_hypothesis(X_train)
            # Calculate error.
            pure_error = A - Y  # (N, C)
            # For Cross-Entropy, sigmoid function cancels its own derivative.
            # Update the weights and biases accordingly.
            self.W = self.W - lr * (
                (1 / N) * X_train.T @ pure_error  # Gradient (D, C)
                + (self.hp.l2_coef * self.W)  # L2 Regularization
            )
            if self.hp.use_bias:
                self.B = self.B - lr * (1 / N) * np.sum(
                    pure_error, axis=0, keepdims=True
                )
            if self.hp.diminishing_lr:
                lr *= self.hp.lr_dim_coef
            completed = _ + 1
            progress_bar.update(
                HTML(
                    '<progress value="{}" max="{}"></progress> {}%'.format(
                        completed,
                        self.hp.niters,
                        int(completed / self.hp.niters * 100),
                    )
                )
            )
        return self.W, self.B

    def _calc_matrix_sigmoid(self, Z: FloatArrayT) -> FloatArrayT:
        """Returns an array of the same shape as Z, where each element is the sigmoid of the corresponding element in Z."""
        sigmoid_val = 1 / (1 + np.exp(-Z))
        return sigmoid_val

    def _compute_hypothesis(self, X: FloatArrayT):
        if self.B is None:
            self.B = 0
        """
        Computes the value of the hypothesis according to the logistic regression rule.
        Y = sigmoid(W*X + b)
        """
        Z = X @ self.W + self.B
        return self._calc_matrix_sigmoid(Z)
