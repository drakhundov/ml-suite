import numpy as np


class DiscreteMAP:
    def __init__(self):
        self.classes = None
        self.P_y = None
        self.P_xy = None
        self.P_xy_log = None
        self.P_y_log = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train)
        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must have the same size")
        self.nsamples = X_train.shape[0]
        self.nfeat = X_train.shape[1]
        # Assign P(y) for each class.
        classes, counts = np.unique(y_train, return_counts=True)
        self.classes = classes
        self.nclasses = len(classes)
        self.P_y = counts / y_train.shape[0]
        # Assign P(x | y) for all features.
        self.P_xy = np.zeros((self.nfeat, self.nclasses))
        for i, c in enumerate(classes):
            # Take data points that belong to class 'c'.
            X_c = X_train[y_train == c]
            feat_tot = np.sum(X_c[:, fno])
            tot = np.sum(X_c)
            if feat_tot == 0 or tot == 0:
                self.P_xy[fno][i] = 0.0001
            for fno in range(self.nfeat):
                # Add 0.0001 to avoid log(0.0)
                self.P_xy[fno][i] = feat_tot / tot
        self.P_xy_log = np.log(self.P_xy)
        self.P_y_log = np.log(self.P_y)

    def predict(self, X_new: np.ndarray):
        log_prob = np.dot(X_new, self.P_xy_log) + self.P_y_log
        return self.classes[np.argmax(log_prob, axis=1)]
