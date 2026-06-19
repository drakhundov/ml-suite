from abc import ABC, abstractmethod
from typing import Union, Tuple, Generic

from sklearn.model_selection import train_test_split

from mlsuite.protocol import SupModelT, FloatArrayT


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
