import numpy as np

class KMCClassifier:
    def __init__(self):
        pass

    def fit(self, X_train, y_train, niter: int = 1000):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
