from DecisionTree import DecisionTree
import numpy as np
from scipy import stats


class RandomForest:

    def __init__(self , n_trees = 100 , n_features = 10 ):
        self.n_trees = n_trees
        self.n_features = n_features
        self.features_indices_tree = []
        self.trees= [DecisionTree() for _ in range(n_trees)]


    def fit(self , X , y):
        for tree in self.trees:
            # split the data on 0 axis (samples axis)
            X_sample , y_sample = self._bootstrap_samples(X , y)

            # split the data on 1 axis (features axis)
            features_indices = np.random.choice(X.shape[1] , self.n_features , replace=False)
            self.features_indices_tree.append(features_indices)
            X_sample = X_sample[:, features_indices]
            tree.fit(X_sample , y_sample)


    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    

    def predict(self , X):

        predictions = stats.mode([tree.predict(X[: , self.features_indices_tree[i]]) for  i ,tree in enumerate(self.trees)] , keepdims=False)
        return predictions[0]