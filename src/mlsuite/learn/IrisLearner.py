from sklearn.datasets import load_iris

from .base_learners import SupervisedBaseLearner


class IrisLearner(SupervisedBaseLearner):
    def load_dataset(self):
        return load_iris()['data'], load_iris()['target']
