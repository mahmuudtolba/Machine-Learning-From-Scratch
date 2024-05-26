import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



class GradientBoostingRegressor:

    def __init__(self , n_estimators = 100 , lr = 0.1 , max_depth = 3):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None


    def fit(self, X , y):
        self.initial_prediction = np.mean(y)
        predictions = np.full(y.shape, self.initial_prediction)


        for i in range(self.n_estimators):
            # compute residuals
            residuals = y - predictions
            
            # fit tree to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X , residuals)
            self.trees.append(tree)

            # Update predictions
            predictions += self.lr * tree.predict(X)




    def predict(self, X):
        predictions = np.full(X.shape[0] , self.initial_prediction)
        for tree in self.trees:
            predictions += self.lr * tree.predict(X)


        return predictions
