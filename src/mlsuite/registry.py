from mlsuite.reg import Regression
from mlsuite.clf import (
    KNNClassifier,
    KMCClassifier,
    DiscreteMAP,
    OvALogisticClassification,
)

models_map = {
    "reg": [Regression],
    "clf": [KNNClassifier, KMCClassifier, DiscreteMAP, OvALogisticClassification],
}
