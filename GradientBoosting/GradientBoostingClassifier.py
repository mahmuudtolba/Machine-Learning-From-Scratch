import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt



class GradientBoostingClassifier:
	def __init__(self , n_estimators = 100 , learning_rate = 0.001 , max_depth = 3):
		self.n_estimators = n_estimators
		self.learning_rate = learning_rate
		self.max_depth = max_depth
		self.models = []

	def fit(self , X , y):
		# That is the log odds which is equal to p / (1 - p)
		self.initial_predictions = np.log(y.mean() / 1 - y.mean())

		for _ in range(self.n_estimators):
			# Here i need to get the residuals so that i would train the tree on it
			residuals = y - self._predict_proba(X)

			stump = DecisionTreeRegressor(max_depth = self.max_depth) # You can add more parameters to that tree
			stump.fit(X , residuals)
			self.models.append(stump)


	def _predict_proba(self , X):
		y_pred = np.full(X.shape[0] , self.initial_predictions)

		for i in range(len(self.models)):
			y_pred += self.models[i].predict(X) * self.learning_rate

		# Return log-odds to probability through sigmoud function

		return 1 / (1 + np.exp(-y_pred))



	def predict(self , X):
		return (self._predict_proba(X) > 0.5).astype(int)


	def predict_proba(self , X):
		return self._predict_proba(X)
  




# Example usage:
# Load some binary classification data
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data
y = (data.target == 1 ).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
