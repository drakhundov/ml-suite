from mlsuite.reg import LinearRegression
from mlsuite.clf import (
    KNNClassifier,
    KMCClassifier,
    DiscreteMAP,
    MultiClassLogisticRegression,
)

models_map = {
    "reg": [LinearRegression],
    "clf": [KNNClassifier, KMCClassifier, DiscreteMAP, MultiClassLogisticRegression],
}
