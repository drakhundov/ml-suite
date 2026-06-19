from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt

FloatArrayT = npt.NDArray[np.floating]
IntArrayT   = npt.NDArray[np.integer]   # Useful for classification labels.

@runtime_checkable
class SupervisedModel(Protocol):
    def fit(self, X_train: FloatArrayT, y_train: FloatArrayT) -> "SupervisedModel": ...
    def predict(self, X_new: FloatArrayT) -> FloatArrayT: ...


@runtime_checkable
class UnsupervisedModel(Protocol):
    def fit(self, X_train: FloatArrayT) -> "UnsupervisedModel": ...
    def predict(self, X_new: FloatArrayT) -> FloatArrayT: ...


SupModelT = TypeVar("ModelT", bound=SupervisedModel)
UnsModelT = TypeVar("ModelT", bound=UnsupervisedModel)
