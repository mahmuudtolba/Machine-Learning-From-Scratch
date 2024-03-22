import numpy as np


X = np.random.randn(10,3)
w = np.random.randn(3,1)
y_pred = np.dot(X , w)

print(y_pred)
print(y_pred.flatten())
