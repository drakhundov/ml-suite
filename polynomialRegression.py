import numpy as np


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def leastSquares(X, Y, degree):
    """
    Input:
    X and Y are two-dim numpy arrays.
    X dims: (N of samples, feature dims)
    Y dim: (N of samples, response dims)
    degree: Degree of the polynomial
    Output:
    Weight (Coefficient) vector
    Vandermonde Matrix
    """
    if degree < 1:
        raise ValueError("Polynomial degree must be greater than or equal to 1.")
    X_poly = np.ones((X.shape[0], 1))
    for d in range(1, degree + 1):
        X_poly = np.c_[X_poly, X**d]
    W = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ Y
    return W, X_poly


class gradientDescent:
    def __init__(self, x, y, w, lr, num_iters):
        self.x = x
        self.y = y
        self.lr = lr
        self.num_iters = num_iters
        self.epsilon = 1e-4
        self.w = w.copy()
        self.weight_history = [self.w]
        self.cost_history = [
            np.sum(np.square(self.predict(self.x) - self.y)) / self.x.shape[0]
        ]

    def gradient(self):
        y_pred = self.predict(self.x)
        gradient = (2 / self.x.shape[0]) * self.x.T @ (y_pred - self.y)
        return gradient

    def fit(self, lr=None, n_iterations=None):
        k = 0
        if n_iterations is None:
            n_iterations = self.num_iters
        if lr != None:
            self.lr = lr

        for k in range(n_iterations):
            grad = self.gradient()
            if self.lr == "diminishing":
                self.lr = 1 / (k + 1)
            self.w -= self.lr * grad
            self.weight_history.append(self.w.copy())
            cost = np.sum(np.square(self.predict(self.x) - self.y)) / self.x.shape[0]
            self.cost_history.append(cost)
            if (
                k > 0
                and abs(self.cost_history[-2] - self.cost_history[-1]) < self.epsilon
            ):
                break
        return self.w, k

    def predict(self, x):
        y_pred = np.zeros_like(self.y)
        y_pred = x.dot(self.w)
        return y_pred
