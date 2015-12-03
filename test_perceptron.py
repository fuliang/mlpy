import pylab as pl
import numpy as np
# from perceptron import Perceptron
from perceptron_sgd import PerceptronSGD

pima = np.loadtxt('data/pima-indians-diabetes.data', delimiter=',')

# indices0 = pima[:, -1] == 0
# indices1 = pima[:, -1] == 1
# pl.plot(pima[indices0, :8], pima[indices0, -1], 'go')
# pl.plot(pima[indices1, :8], pima[indices1, -1], 'rx')
# pl.show()

p = PerceptronSGD(1000, 0.01)
# p = Perceptron(1000, 0.05)
p.fit(pima[:, :8], pima[:,8:9])
y = p.predict(pima[:, :8])
print y
# print  pima[:, 8:9]
print np.sum(pima[:, 8:9].flatten() == y) * 100.0 / len(y)

