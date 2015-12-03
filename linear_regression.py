class LinearRegression(object):

    def fit(X, y):
        n_sample, n_feature = np.shape(X)
        X = np.concatenate((inputs, np.ones((n_sample, 1))), axis=1)
        X_t_X_inv = np.linalg.inv(np.dot(np.transpose(inputs), inputs))
        self.weights = np.dot(np.dot(X_t_X_inv, np.transpose(X)), y)

    def predict(X):
        n_sample, n_feature = np.shape(X)
        X = np.concatenate((inputs, np.ones((n_sample, 1))), axis=1)
        return np.dot(X, self.weights)

if __name__ == '__main__':

