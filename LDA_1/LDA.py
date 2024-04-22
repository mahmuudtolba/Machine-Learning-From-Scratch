import numpy as np

class LinearDiscriminantAnalysis:
    def __init__(self):
        self.mean_class0 = None
        self.mean_class1 = None
        self.var_shared = None
        self.prior_class0 = None
        self.prior_class1 = None
    
    def fit(self, X, y):
        X_class0 = X[y == 0]
        X_class1 = X[y == 1]
        
        self.mean_class0 = np.mean(X_class0)
        self.mean_class1 = np.mean(X_class1)
        
        self.var_shared = np.var(np.concatenate((X_class0, X_class1)))
        
        self.prior_class0 = len(X_class0) / len(X)
        self.prior_class1 = len(X_class1) / len(X)
        
    def predict(self, X):
        y_pred = []
        for x in X:
            # Calculate class probabilities using Bayes' theorem
            p_class0 = (1 / np.sqrt(2 * np.pi * self.var_shared)) * np.exp(-(x - self.mean_class0)**2 / (2 * self.var_shared)) * self.prior_class0
                       
            p_class1 = (1 / np.sqrt(2 * np.pi * self.var_shared)) *np.exp(-(x - self.mean_class1)**2 / (2 * self.var_shared)) * self.prior_class1


            # Predict the class with higher probability
            if p_class0 > p_class1:
                y_pred.append(0)
            else:
                y_pred.append(1)
        
        return np.array(y_pred)
