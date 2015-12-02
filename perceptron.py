import numpy as np

class Perceptron(object):

    def __init__(self, n_iter, eta):
        self.n_iter = n_iter
        self.eta = eta
        self.weights = None

    def fit(self, X, y):
        n_sample, n_features = X.shape
        if np.ndim(y) > 1:
            self.n_out = np.shape(y)[1]
        else:
            self.n_out = 1

        self.weights = np.random.rand(n_features + 1, self.n_out)
        X = np.concatenate((X, np.ones((n_sample, 1))), axis=1)
        
        for i in range(self.n_iter):
            self.activations = self._forward(X)
            # print self.activations
            self.weights -= self.eta * np.dot(np.transpose(X), self.activations - y)

    def predict(self, X):
        n_sample, n_features = X.shape
        X = np.concatenate((X, np.ones((n_sample, 1))), axis=1)
        y = np.dot(X, self.weights)
        return np.where(y > 0, 1, 0)
        

    def _forward(self, inputs):
        activations = np.dot(inputs, self.weights)
        return np.where(activations > 0, 1, 0)

if __name__ == '__main__':
    classifier = Perceptron(10, 0.25)
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

