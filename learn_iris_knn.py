from sklearn.datasets import load_iris

from KNN_Classifier import KNN_Classifier

# Load sample data
iris = load_iris()
X, y = iris.data, iris.target

# Split into train/test
n_train = 120
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Fit and predict
clf = KNN_Classifier(k=3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Check accuracy
accuracy = sum(predictions == y_test) / len(y_test)
print(f"Accuracy: {accuracy:.2%}")
