from itertools import combinations_with_replacement

import numpy as np

from mlsuite.protocol import FloatArrayT
from .Conf import GDConfig, LSConfig, OptimizationMethod

# TODO: implement stochastic gradient descent


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
            self.W = W_start
        else:
            self.W = np.zeros((D, 1))

        prev_cost = float("inf")
        lr = self.hp.lr
        for iterno in range(self.hp.niters):
            grad = self._calc_gradient(X_train, y_train)
            l2_reg = 0
            if self.hp.l2_coef != 0.0:
                l2_reg += 2 * self.hp.l2_coef * self.W
                if self.hp.use_bias:
                    l2_reg[0, 0] = 0.0
            self.W -= lr * (grad + l2_reg)
            cost = np.sum(
                (X_train.dot(self.W) - y_train) ** 2
            ) / N + self.hp.l2_coef * np.sum(self.W**2)
            if self.hp.epsilon is not None:
                # Break once error < epsilon.
                if iterno > 0 and abs(prev_cost - cost) < self.hp.epsilon:
                    break
                else:
                    prev_cost = cost
            if self.hp.diminishing_lr:
                lr *= self.hp.lr_dim_coef
        return self.W

    def _calc_gradient(self, X: FloatArrayT, y: FloatArrayT):
        y_pred = X @ self.W
        return (2 / X.shape[0]) * (X.T @ (y_pred - y))


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


class Regression:
    def __init__(
        self,
        optim_method: OptimizationMethod,
        *,
        gdconf: GDConfig | None = None,
        lsconf: LSConfig | None = None,
    ):
        self.optim_method = optim_method
        # Set hyperparameters.
        if self.optim_method == OptimizationMethod.GRAD:
            self.solver = GDSolver(gdconf)
            self.hp = gdconf
        else:
            self.solver = LSSolver(lsconf)
            self.hp = lsconf
        if self.hp == None:
            raise ValueError("Must provide configuration via GDConfig or LSConfig")
        self.X_train = None
        self.y_train = None
        self.W = None
        self.bias = None
        self.coef = None

    def fit(self, X_train: FloatArrayT, y_train: FloatArrayT):
        """
        shape(X_train) = (# of data points, # of features)
        shape(y_train) = (# of data points, 1)
        shape(W)       = (# of features, 1)
        """
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must have same number of columns")
        # Training data is saved for reference.
        self.X_train = X_train
        self.y_train = y_train
        if self.hp.normalize_data:
            mu = X_train.mean(axis=0, keepdims=True)
            sigma = X_train.std(axis=0, keepdims=True) + 1e-8
            X_train = (X_train - mu) / sigma
            # Save normalization coefficients to apply to test set.
            self.train_mu = mu
            self.train_std = sigma
        self.W = self.solver.fit(self._design_matrix(X_train), y_train)
        if self.hp.use_bias:
            self.bias = float(np.asarray(self.W).reshape(-1)[0])
            self.coef = self.W[1:]
        else:
            self.bias = None
            self.coef = None

    def predict(self, X_new: FloatArrayT) -> FloatArrayT:
        if self.W is None:
            raise ValueError("Must first train the model before predicting")
        elif len(X_new) == 0:
            raise ValueError("Must provide data for prediction")
        if self.hp.normalize_data:
            X_new = (X_new - self.train_mu) / self.train_std
        X_new = self._design_matrix(X_new)
        return X_new.dot(self.W)

    def _design_matrix(self, X: FloatArrayT) -> FloatArrayT:
        """Designs training matrix based on configuration (bias, polynomial)."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N,D), got {X.shape}")
        N, D = X.shape

        columns = []

        # Add bias column.
        if self.hp.use_bias:
            columns.append(np.ones((N, 1)))

        for deg in range(1, self.hp.degree + 1):
            # combinations_with_replacement(range(D), 3) yields (0,0,0), (0,0,1), (0,1,2), etc.
            for comb in combinations_with_replacement(range(D), deg):
                # Take the product along the columns for the chosen indices.
                new_col = np.prod(X[:, comb], axis=1, keepdims=True)
                columns.append(new_col)

        return np.hstack(columns)
