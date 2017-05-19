# Based on tutorial https://www.youtube.com/watch?v=N4gDikiec8E&index=7&list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3

import numpy
import matplotlib.pyplot as plt

epochs = 2000

x = numpy.array([[0,0,1],
		 [0,1,1],
		 [1,0,1],
		 [1,1,1]])

y = numpy.array([[0],
		 [1],
		 [1],
		 [0]])

print x
print y

syn0 = 2 * numpy.random.random((3, 4)) - 1
syn1 = 2 * numpy.random.random((4, 1)) - 1

print syn0
print syn1

def nonlin(x, deriv=False):
	if deriv == True:
		return x * (1 - x)

	return 1 / (1 + numpy.exp(-x))

scale = epochs * 0.01
errors = []

for i in xrange(epochs):
	l0 = x
	l1 = nonlin(numpy.dot(l0, syn0))
	l2 = nonlin(numpy.dot(l1, syn1))

	l2_error = y - l2
	l2_delta = l2_error * nonlin(l2, deriv=True)

	if (i % scale) == 0:
		errors.append(numpy.mean(numpy.abs(l2_error)))
		print "Error: " + str(errors[-1])

	l1_error = l2_delta.dot(syn1.T)
	l1_delta = l1_error * nonlin(l1, deriv=True)

	syn1 += l1.T.dot(l2_delta)
	syn0 += l0.T.dot(l1_delta)

plt.scatter(xrange((int)(epochs / scale)), errors)
plt.show()
