import numpy as np
from collections import Counter


class KNNClassifier:
    def __init__(self, k: int=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Lazy learner."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def predict(self, X_new: np.ndarray):
        # X_new -> column vector
        # self.x_train -> feature matrix
        dist2 = np.sum(X_new**2, axis=1)[:, np.newaxis] + np.sum(self.X_train**2, axis=1) - 2 * np.dot(X_new, self.X_train.T)
        #* No need to take the square root.
        # distances = np.sqrt(np.maximum(dists, 0))

        predictions = []
        for row in dist2:
            k_indices = np.argsort(row)[:self.k]
            
            k_labels = self.y_train[k_indices]
            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)
            
        return predictions
