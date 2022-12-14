import numpy as np
from collections import Counter


class KNN:
  def __init__(self, k):
    self.k = k
    #self.metric = metric
    
  def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y

  def predict(self, X):
    predictions = [self._predict(x) for x in X]
    return predictions

  def _predict(self, x):
    distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    k_indices = np.argsort(distances)[:self.k]
    k_neareset_labels = [self.y_train[i] for i in k_indices]
    most_common= Counter(k_neareset_labels).most_common()

    return most_common[0][0]

#test
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)


knn = KNN(k = 3)
knn.fit(X_train, y_train)
pred = knn.predict( X = X_test)

print(accuracy_score(pred, y_test))
