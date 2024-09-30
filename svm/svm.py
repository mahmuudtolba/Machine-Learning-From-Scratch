import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split



class LinearSVM:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w = None  # Weight vector
        self.b = None  # Bias term

    def fit(self, X, y):
        """
        Fit the SVM model to the training data X with labels y.
        
        X: Input data, shape (n_samples, n_features)
        y: Labels, shape (n_samples,)
        """
        n_samples, n_features = X.shape
        # Initialize the weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Training with gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Check if the constraint is violated (misclassification)
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) < 1
                
                if condition:  # If misclassified
                    # Update weights and bias
                    self.w -= self.lr * (self.w - y[idx] * x_i)
                    self.b -= self.lr * (-y[idx])
                else:
                    # Update weights only for the regularization term
                    self.w -= self.lr * self.w

    def predict(self, X):
        """
        Make predictions using the learned weights and bias.
        
        X: Input data, shape (n_samples, n_features)
        """
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)

# Test the Linear SVM implementation


X, y = datasets.make_blobs(
        n_samples=3000, n_features=12, centers=15, cluster_std=1.05, random_state=40
    )
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

clf = LinearSVM()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def accuracy(y_true, y_pred):
  accuracy = np.sum(y_true == y_pred) / len(y_true)
  return accuracy

print("SVM classification accuracy", accuracy(y_test, predictions))
