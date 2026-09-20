from mlsuite.reg import Regression
from mlsuite.clf import (
    KNNClassifier,
    KMCClassifier,
    MultimodalNaiveBayesianClassification,
    OvALogisticClassification,
)

models_map = {
    "reg": [Regression],
    "clf": [KNNClassifier, KMCClassifier, MultimodalNaiveBayesianClassification, OvALogisticClassification],
}
