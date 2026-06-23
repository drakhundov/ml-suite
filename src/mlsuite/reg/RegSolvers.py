import numpy as np

from mlsuite.protocol import FloatArrayT
from .Conf import LSConfig, GDConfig


class GDSolver:
    def __init__(self, conf: GDConfig):
        self.hp = conf

    def fit(
        self,
        X_train: FloatArrayT,
        y_train: FloatArrayT,
        W_start: FloatArrayT | None = None,
    ) -> FloatArrayT:
        """
        Solves linear regression using Gradient Descent with given hyperparameters.
        Returns resulting weights.
        """
        N, D = X_train.shape

        y_train = np.asarray(y_train)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        elif y_train.ndim == 2 and y_train.shape[1] != 1:
            raise ValueError(f"y_train must be (N,1); got {y_train.shape}")
        if y_train.shape[0] != N:
            raise ValueError("Number of labels doesn't match number of data points")

        if W_start is not None:
            if W_start.shape[0] != D:
                raise ValueError(
                    "Initial weights' shape does not match training data shape"
                )
            W = W_start
        else:
            W = np.zeros((D, 1))

        def _calc_gradient():
            nonlocal W, X_train, y_train
            y_pred = X_train @ W
            return (2 / X_train.shape[0]) * (X_train.T @ (y_pred - y_train))

        prev_cost = float("inf")
        lr = self.hp.lr
        for iterno in range(self.hp.niters):
            grad = _calc_gradient()
            l2_reg = 0
            if self.hp.l2_coef != 0.0:
                l2_reg += 2 * self.hp.l2_coef * W
                if self.hp.use_bias:
                    l2_reg[0, 0] = 0.0
            W -= lr * (grad + l2_reg)
            cost = np.sum(
                (X_train.dot(W) - y_train) ** 2
            ) / N + self.hp.l2_coef * np.sum(W**2)
            if self.hp.epsilon is not None:
                # Break once error < epsilon.
                if iterno > 0 and abs(prev_cost - cost) < self.hp.epsilon:
                    break
                else:
                    prev_cost = cost
            if self.hp.diminishing_lr:
                lr *= self.hp.lr_dim_coef
        return W


class LSSolver:
    def __init__(self, conf: LSConfig):
        self.hp = conf  # Hyperparameters.

    def fit(self, X_train: FloatArrayT, y_train: FloatArrayT) -> FloatArrayT:
        """
        Solves linear regression using Least Squares with given hyperparameters.
        Returns resulting weights.
        """
        A = np.dot(X_train.T, X_train)
        if self.hp.l2_coef != 0.0:
            I_reg = np.eye(X_train.shape[1])
            if self.hp.use_bias:
                # Don't penalyze bias.
                I_reg[0, 0] = 0.0
            A += self.hp.l2_coef * I_reg
        W = np.linalg.inv(A) @ X_train.T @ y_train
        return W
