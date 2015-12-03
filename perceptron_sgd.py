import numpy as np

class PerceptronSGD(object):

    def __init__(self, n_iter=10, eta=0.01):
        self.n_iter = n_iter
        self.eta = eta

    def fit(self, X, y):
        n_sample, n_feature = X.shape
        X = np.concatenate((X, np.ones((n_sample, 1))), axis=1)
        self.weights = np.zeros((1 + n_feature))
        for _ in range(self.n_iter):
            for xi, target in zip(X, y):
                self.weights += self.eta * (target - self._forward(xi)) * xi
    
    def _forward(self, X):
        y = np.dot(X, self.weights)
        return np.where(y > 0.0, 1, 0)

    def predict(self, X):
        n_sample, n_feature = X.shape
        X = np.concatenate((X, np.ones((n_sample, 1))), axis=1)
        y = np.dot(X, self.weights)
        return np.where(y > 0.0, 1, 0)

if __name__ == '__main__':
    classifier = PerceptronSGD(10, 0.25)
    X = np.array([
       [0, 0],
       [0, 1],
       [1, 0],
       [1, 1]])
    y = np.array([[0], [0], [0], [1]])
    # y = np.array([[0, 1], [0, 1], [0, 1], [1, 0]])
    classifier.fit(X, y)
    print classifier.weights
    print classifier.predict(X)

