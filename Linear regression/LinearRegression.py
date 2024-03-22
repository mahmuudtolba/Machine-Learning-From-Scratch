import numpy as np


class LinearRegression:
    def __init__(self , lr = 0.01 , n_iterations = 1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None


    def fit(self , X , y):
        n_samples , n_feats = X.shape

        # Initialize weights and bias
        self.weights = np.random.randn(n_feats , 1)
        self.bias = 0


        for _ in range(self.n_iterations):
            y_predicted = np.dot(X , self.weights).flatten() + self.bias

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T , (y_predicted - y).reshape(-1,1))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X):
        y_predicted = np.dot(X , self.weights).flatten() + self.bias
        return y_predicted



