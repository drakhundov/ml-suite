from abc import ABC, abstractmethod
from typing import Union, Tuple, Generic

from sklearn.model_selection import train_test_split

from mlsuite.protocol import SupModelT, UnsModelT, FloatArrayT
from mlsuite.registry import models_map
from .TestStats import TestStats, produce_test_stats_reg, produce_test_stats_clf


class SupervisedBaseLearner(ABC, Generic[SupModelT]):
    def __init__(
        self,
        model: SupModelT,
        test_ratio: float = 0.2,
        split_by_sk: bool = True,
        random_state: int = 42,
    ):
        self.model = model
        self.test_sz = test_ratio
        self.split_by_sk = split_by_sk
        self.random_state = random_state

        X, y = self.load_dataset()  # <-- subclass should provide this
        self.X_train, self.X_test, self.y_train, self.y_test = self.split(X, y)

    @abstractmethod
    def load_dataset(self) -> Tuple[FloatArrayT, FloatArrayT]:
        """Returns X, y."""
        raise NotImplementedError

    def split(self, X: FloatArrayT, y: FloatArrayT):
        if self.split_by_sk:
            return train_test_split(
                X, y, test_size=self.test_sz, random_state=self.random_state
            )
        n = X.shape[0]
        n_train = int((1 - self.test_sz) * n)
        return X[:n_train], X[n_train:], y[:n_train], y[n_train:]

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def test(self) -> TestStats:
        pred = self.model.predict(self.X_test)
        if self.model.__class__ in models_map["reg"]:
            return produce_test_stats_reg(pred, self.y_test)
        else:
            return produce_test_stats_clf(pred, self.y_test)

class UnsupervisedBaseLearner(ABC, Generic[UnsModelT]):
    def __init__(
        self,
        model: UnsModelT,
        test_ratio: float = 0.2,
        split_by_sk: bool = True,
        random_state: int = 42,
    ):
        self.model = model
        self.test_sz = test_ratio
        self.split_by_sk = split_by_sk
        self.random_state = random_state

        X = self.load_dataset()  # <-- subclass should provide this
        self.X_train, self.X_test = self.split(X)

    @abstractmethod
    def load_dataset(self) -> Tuple[FloatArrayT, FloatArrayT]:
        """Returns X."""
        raise NotImplementedError

    @abstractmethod
    def load_labels(self, FloatArrayT):
        """Returns y."""
        raise NotImplementedError

    def split(self, X: FloatArrayT):
        if self.split_by_sk:
            return train_test_split(
                X, test_size=self.test_sz, random_state=self.random_state
            )
        n = X.shape[0]
        n_train = int((1 - self.test_sz) * n)
        return X[:n_train], X[n_train:]

    def fit(self):
        self.model.fit(self.X_train)

    def test(self) -> TestStats:
        _, y_test = self.split(self.load_labels())
        pred = self.model.predict(self.X_test)
        # Regression is supervised, so no need to check for regression.
        return produce_test_stats_clf(pred, y_test)
