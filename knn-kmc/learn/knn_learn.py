import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clf import KNNClassifier
from custom_learn import IrisLearner

clf = KNNClassifier(k=5)
learner = IrisLearner(clf, test_sz=0.2, split_by_sk=True)
stats = learner.fit_and_get_stats()

for _class in range(3):
    print(f"Class {_class} accuracy: {stats[_class].accuracy*100:.2f}")
