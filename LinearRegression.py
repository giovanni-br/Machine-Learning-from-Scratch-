import numpy as np

class Linear_Regression:

  def __init__(self, n_it=1000, lr = 0.001):
    self.lr = lr
    self.n_it = n_it
    self.weights = None
    self.bias = None

  def fit(self, X, y):
    n_samples, n_features = X.shape
    self.bias =0
    self.weights = np.zeros( n_features)

    #gradient descent
    for i in range(self.n_it):
      y_predicted = np.dot(X, self.weights) + self.bias
      dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
      db = (1/n_samples) * np.sum((y_predicted - y))
      self.weights -= self.lr*dw
      self.bias -= self.lr*db  

  def predict(self, X):
    y_approximated = np.dot(X, self.weights) + self.bias
    return  y_approximated
