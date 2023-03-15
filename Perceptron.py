import numpy as np

#A Perceptron, the first Neural Network, that solves only linearly separated problems 

class Perceptron:

  def __init__(self, learning_rate = 0.1, n_iters = 1000):
    self.lr = learning_rate
    self.n = n_iters
    self.activation = self.heaviside
    self.bias = None
    self.weights = None
  
  def heaviside(self, x):
    return np.where(x>=0, 1 ,0)

  def fit(self, X, y):
    n_samples, n_features = X.shape

    self.weights = np.zeros(n_features)
    self.bias = 0

    y_2 = [0 if i<=0 else 1 for i in y]

    for _ in range(self.n):
      for index, x_i in enumerate(X):
        linear_out = np.dot(x_i, self.weights) + self.bias
        y_predicted = self.heaviside(linear_out) 

        new = self.lr*(y_2[index] - y_predicted)

        self.weights += new*x_i
        self.bias += new


  def predict(self, X):
    linear_out = np.dot(X, self.weights) + self.bias
    y_predicted = self.heaviside(linear_out)
    return y_predicted



from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X, y = make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

perceptron = Perceptron(learning_rate = 0.01, n_iters = 1000)
perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)

print(accuracy_score(y_pred, y_test))
