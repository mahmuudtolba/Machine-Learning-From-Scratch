from sklearn import datasets
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTree
import numpy as np


data = datasets.load_breast_cancer()
X , y = data.data , data.target




X_train , X_test , y_train , y_test = train_test_split(
    X ,y  , test_size= 0.2
)

clf = DecisionTree(max_depth=5)
clf.fit(X=X_train , y=y_train)

prediction = clf.predict(X_test)

def accuracy(y_test , y_pred):
    return np.sum(y_test == y_pred) / len(y_test)



acc= accuracy(y_test , prediction)
print(acc)