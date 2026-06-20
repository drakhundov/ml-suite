import numpy as np

from mlsuite.protocol import FloatArrayT


class KMCClassifier:
    def __init__(self, k: int = 5, niter: int = 100, threshold: float = 1e-4):
        self.k = k
        self.niter = niter
        self.threshold = threshold

    def fit(self, X_train: FloatArrayT):
        X_train = np.array(X_train)
        n_samples, n_features = X_train.shape

        # Initialize centroids randomly by picking k points from the dataset
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X_train[random_indices]

        for iteration in range(self.niter):
            # dists[p_i][c_j] -> Euclidean distance from point i (from X) to centroid j
            dists = (
                    np.sum(X_train ** 2, axis=1)[:, np.newaxis]
                    + np.sum(self.centroids ** 2, axis=1)
                    - 2 * np.dot(X_train, self.centroids.T)
            )
            # labels[j] -> index of the centroid closest to point j
            labels = np.argmin(dists, axis=1)

            new_centroids = np.zeros((self.k, n_features))
            for i in range(self.k):
                points_in_cluster = X_train[labels == i]
                if len(points_in_cluster) > 0:
                    new_centroids[i] = np.mean(points_in_cluster, axis=0)
                else:
                    random_idx = np.random.choice(n_samples)
                    new_centroids[i] = X_train[random_idx]
                    print(f"Centroid {i} was replaced with data point at {random_idx}.")

            centroid_shift = np.sum(
                np.sqrt(np.sum((self.centroids - new_centroids) ** 2, axis=1))
            )
            self.centroids = new_centroids

            if centroid_shift < self.threshold:
                print(f"Converged early at iteration {iteration}")
                break

    def predict(self, X: FloatArrayT) -> FloatArrayT:
        X = np.array(X)
        dists = (
            np.sum(X**2, axis=1)[:, np.newaxis]
            + np.sum(self.centroids**2, axis=1)
            - 2 * np.dot(X, self.centroids.T)
        )
        return np.argmin(dists, axis=1)
