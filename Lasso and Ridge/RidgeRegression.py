import numpy as np


class RidgeRegression:
    def __init__(self, alpha=0.5, lr=0.01, n_iterations=10000):
        self.alpha = alpha  # Regularization parameter
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
            # [3 , 10] . [10 , 1] -> [3 , 1] + [3 , 1] --> [3 , 1]
            dw = (1 / n_samples) *  ( np.dot(X.T , (y_predicted - y).reshape(-1,1)) + 2 * self.alpha * self.weights)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X):
        y_predicted = np.dot(X , self.weights).flatten() + self.bias
        return y_predicted



