import numpy as np
from typing import List, Tuple


class MultiClassLogisticRegression:
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ):
        """
        x_train: a matrix in which each column contains an instance.
        y_train: a matrix that contains one integer for each instance, indicating the instance's label.
        x_test, y_test: test datasets with similar structure.

        Note: Classification starts with 0 and goes incrementally.
        """
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self.nfeat = x_train.shape[0]  # Number of features.
        self._m = x_train.shape[1]  # Size of the dataset.
        self.nclass = np.max(y_train) + 1  # Number of classes.
        # Train a separate binary classification model for each class.
        # Put them all into a single matrix for efficiency.
        self._W = np.random.randn(self.nclass, self.nfeat) * 0.01
        self._B = np.zeros((self.nclass, 1))

        self._alpha = 0.05

        self._Y = np.zeros((self.nclass, self._m))
        self._Y[self._y_train, np.arange(self._m)] = 1

    def sigmoid(self, Z: np.ndarray) -> np.ndarray:
        """
        Args:
        - Z: The input array.

        Returns:
        - numpy.ndarray: An array of the same shape as Z, where each element is the sigmoid of the corresponding element in Z.
        """
        sigmoid_val = 1 / (1 + np.exp(-Z))
        return sigmoid_val

    def derivative_sigmoid(self, A: np.ndarray) -> np.ndarray:
        """
        Args:
        - A: The input array. It should be the output of sigmoid(Z).

        Returns:
        - An array of the same shape as A, where each element is the derivative of sigmoid(Z), where A=sigmoid(Z).
        """
        deriv_sigmoid_val = A * (1 - A)
        return deriv_sigmoid_val

    def h_theta(self, X: np.ndarray):
        """
        Computes the value of the hypothesis according to the logistic regression rule.

        Args:
        - X: The input feature matrix.

        Returns:
        - A column vector of predicted values obtained
        """
        Z = self._W @ X + self._B
        h_theta = self.sigmoid(Z)
        return h_theta

    def get_weights_for_digit(self, digit: int) -> np.ndarray:
        """
        Args:
        - digit: The digit for which the weights are to be returned.

        Returns:
        - A row vector of weights from the weights matrix corresponding to the given digit.
        """
        w = self._W[digit, :]
        return w

    def train_mse_loss(self, niters: int) -> Tuple[List, List]:
        """
        Performs a number of iterations using Gradient Descent with MSE.

        Returns a list with the percentage of instances classified correctly in the training and in the test sets.
        """
        classified_correctly_train_list = []
        classified_correctly_test_list = []

        for i in range(niters):
            # Compute hypothesis.
            A = self.h_theta(self._x_train)
            # Calculate error.
            pure_error = A - self._Y
            # For MSE, gradient includes the derivative of sigmoid.
            grad_sig = pure_error * self.derivative_sigmoid(A)
            # Update the weights and biases accordingly.
            self._W = self._W - self._alpha * (1 / self._m) * (
                grad_sig @ self._x_train.T
            )
            self._B = self._B - self._alpha * (1 / self._m) * np.sum(
                grad_sig, axis=1, keepdims=True
            )

            if i % 100 == 0:
                train_preds = np.argmax(A, axis=0)
                classified_correctly = np.sum(train_preds == self._y_train)
                percentage_classified_correctly = classified_correctly * 100 / self._m
                classified_correctly_train_list.append(percentage_classified_correctly)
                images_test = self._x_test
                Y_hat_test = self.h_theta(images_test)
                test_preds = np.argmax(Y_hat_test, axis=0)
                test_correct = np.sum(test_preds == self._y_test)
                classified_correctly_test_list.append(
                    (test_correct) / len(self._y_test) * 100
                )

                print(f"Accuracy[iter {i}]: %.2f" % percentage_classified_correctly)
        return classified_correctly_train_list, classified_correctly_test_list

    def train_cross_entropy_loss(self, niters: int) -> Tuple[List, List]:
        """
        Performs a number of iterations using Gradient Descent with Cross-Entropy + L2 Regularization.

        Returns a list with the percentage of instances classified correctly in the training and in the test sets.
        """

        classified_correctly_train_list_ce = []
        classified_correctly_test_list_ce = []

        for i in range(niters):
            # Compute hypothesis.
            A = self.h_theta(self._x_train)
            # Calculate error.
            pure_error = A - self._Y
            # For Cross-Entropy, sigmoid function cancels its own derivative.
            # Update the weights and biases accordingly.
            self._W = self._W - self._alpha * (
                (1 / self._m) * pure_error @ self._x_train.T  # Gradient
                + (0.02 * self._W)  # L2 Regularization
            )
            self._B = self._B - self._alpha * (1 / self._m) * np.sum(
                pure_error, axis=1, keepdims=True
            )

            if i % 100 == 0:
                train_preds = np.argmax(A, axis=0)
                classified_correctly = np.sum(train_preds == self._y_train)
                percentage_classified_correctly_trainset = (
                    classified_correctly * 100 / self._m
                )
                classified_correctly_train_list_ce.append(
                    percentage_classified_correctly_trainset
                )
                images_test = self._x_test
                Y_hat_test = self.h_theta(images_test)
                test_preds = np.argmax(Y_hat_test, axis=0)
                test_correct = np.sum(test_preds == self._y_test)
                percentage_classified_correctly_testset = (test_correct) / len(self._y_test) * 100
                classified_correctly_test_list_ce.append(
                    percentage_classified_correctly_testset
                )

                print(f"Accuracy[trainset, iter {i}]: %.2f" % percentage_classified_correctly_trainset)
                print(f"Accuracy[testset, iter {i}]: %.2f" % percentage_classified_correctly_testset)

        return classified_correctly_train_list_ce, classified_correctly_test_list_ce
