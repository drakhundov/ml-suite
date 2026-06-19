from sklearn.datasets import fetch_california_housing, load_iris

from .base_learners import SupervisedBaseLearner


class CalHouseLearner(SupervisedBaseLearner):
    def load_dataset(self):
        return fetch_california_housing(data_home='./skdata', return_X_y=True)
