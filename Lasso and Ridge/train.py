import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from RidgeRegression import RidgeRegression
from LinearRegression import LinearRegression
from LassoRegression import LassoRegression

X, y = datasets.make_regression(n_samples=100000, n_features=10, noise=100, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

reg_ = LinearRegression(lr=0.001)
reg_R = RidgeRegression(lr=0.001 , alpha=20)
reg_L = LassoRegression(lr=0.001 , alpha=20)


reg_.fit(X_train,y_train)
reg_R.fit(X_train,y_train)
reg_L.fit(X_train,y_train)


predictions_ = reg_.predict(X_test)
predictions_R = reg_R.predict(X_test)
predictions_L = reg_L.predict(X_test)


def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)

mse_ = mse(y_test, predictions_)
mse_R = mse(y_test, predictions_R)
mse_L = mse(y_test, predictions_L)


print(f'MSE_ : {mse_}')
print(f'MSE_R : {mse_R}')
print(f'MSE_L : {mse_L}')


