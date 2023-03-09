import numpy as np

def gradient_descent(gradient, start,  learn_rate, n_iter=50, tolerance=1e-06):
    vector = start
    for i in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if np.abs(diff) <= tolerance:
            return vector
        vector += diff
