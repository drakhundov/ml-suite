from mlsuite.reg import Regression
from mlsuite.clf import (
    KNNClassifier,
    KMCClassifier,
    DiscreteMAP,
    MultiClassLogisticRegression,
)

models_map = {
    "reg": [Regression],
    "clf": [KNNClassifier, KMCClassifier, DiscreteMAP, MultiClassLogisticRegression],
}
