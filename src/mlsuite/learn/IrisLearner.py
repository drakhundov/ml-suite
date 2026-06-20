from sklearn.datasets import load_iris

from mlsuite.protocol import FloatArrayT
from .base_learners import SupervisedBaseLearner, UnsupervisedBaseLearner


class SupervisedIrisLearner(SupervisedBaseLearner):
    def load_dataset(self):
        return load_iris()['data'], load_iris()['target']


class UnsupervisedIrisLearner(UnsupervisedBaseLearner):
    def load_dataset(self):
        return load_iris()['data']
    
    def load_labels(self) -> FloatArrayT:
        return load_iris()['target']
