from sklearn.datasets import fetch_california_housing

from .base_learners import SupervisedBaseLearner, UnsupervisedBaseLearner


class SupervisedCalHouseLearner(SupervisedBaseLearner):
    def load_dataset(self):
        return fetch_california_housing(data_home='./skdata', return_X_y=True)


class UnsupervisedCalHouseLearner(UnsupervisedBaseLearner):
    def load_dataset(self):
        return fetch_california_housing(data_home='./skdata')['data']
    
    def load_labels(self):
        return fetch_california_housing(data_home='./skdata')['target']
