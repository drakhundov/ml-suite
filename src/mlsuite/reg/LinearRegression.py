from enum import Enum

import numpy as np

from mlsuite.reg import GDConfig, LSConfig

# TODO: implement stochastic gradient descent


class OptimizationMethod(Enum):
    GRAD = 1  # Gradient Descent
    LSQR = 2  # Least Squares


class GDSolver:
    def __init__(self, conf: GDConfig):
        self.hp = conf

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        W_start: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Solves linear regression using Gradient Descent with given hyperparameters.
        Returns resulting weights.
        """
        N, D = X_train.shape

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
        for iterno in range(self.hp.num_iters):
            grad = self._calc_gradient(X_train, y_train)
            l2_reg = 0
            if self.hp.l2_coef != 0.0:
                l2_reg += 2 * self.hp.l2_coef * self.W
            lr = self.hp.lr
            if self.hp.diminishing_lr:
                lr /= iterno + 1
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
        return self.W

    def _calc_gradient(self, X: np.ndarray, y: np.ndarray):
        y_pred = X @ self.W
        return (2 / X.shape[0]) * (X.T @ (y_pred - y))


class LSSolver:
    def __init__(self, conf: LSConfig):
        self.hp = conf  # Hyperparameters.

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """
        Solves linear regression using Least Squares with given hyperparameters.
        Returns resulting weights.
        """
        # Add bias column if bias is toggled on.
        if self.hp.bias:
            X = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        else:
            X = X_train
        A = np.dot(X.T, X)
        if self.hp.l2_coef != 0.0:
            I_reg = np.eye(X.shape[1])
            if self.hp.bias:
                # Don't penalyze bias.
                I_reg[0, 0] = 0.0
            A += self.hp.l2_coef * I_reg
        W = np.linalg.inv(A) @ X.T @ y_train
        return W


class LinearRegression:
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

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
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
        self.W = self.solver.fit(X_train, y_train)
        if self.hp.bias:
            self.bias = (
                self.W[0].item() if isinstance(self.W, np.ndarray) else self.W[0]
            )
            self.coef = self.W[1:]
        else:
            self.bias = None
            self.coef = None

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise ValueError("Must first train the model before predicting")
        elif len(X_new) == 0:
            raise ValueError("Must provide data for prediction")
        return X_new.dot(self.W)
