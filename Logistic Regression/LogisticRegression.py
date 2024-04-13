import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_feats = X.shape

        # Initialize weights and bias
        # [3 , 1]
        self.weights = np.random.randn(n_feats, 1)
        self.bias = 0

        for _ in range(self.n_iterations):
            # Forward pass
            # [10,3] . [3 , 1] --> [10 , 1]
            linear_output = np.dot(X, self.weights).flatten() + self.bias
            y_predicted = self.sigmoid(linear_output)

            # Compute gradients
            # [3 , 10] . [10 , 1] --> [3 , 1]
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y).reshape(-1 , 1))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            #print(f'dw :{dw.shape}')
            #print(f'weights :{self.weights.shape}')

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_output = np.dot(X, self.weights).flatten() + self.bias
        y_predicted = self.sigmoid(linear_output)
        return y_predicted
