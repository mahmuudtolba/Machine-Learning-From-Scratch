from DecisionTree import DecisionTree
import numpy as np
from scipy import stats

class BaggingTree:


    def __init__(self , n_trees):
        self.n_trees = n_trees
        self.trees = [DecisionTree() for i in range(n_trees)]

    def fit(self , X , y):
        for tree in self.trees:
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    
    def predict(self , X):
        predictions = stats.mode([tree.predict(X) for tree in self.trees] , keepdims=False)
        return predictions[0]