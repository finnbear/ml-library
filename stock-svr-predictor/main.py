# Based on tutorial https://www.youtube.com/watch?v=SSu00IRRraY

import csv
import numpy
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

def load_data(path):
	with open(path, 'r') as data_file:
		data_reader = csv.reader(data_file)
		next(data_reader)
		for row in data_reader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return

def predict_price(input):
	x = numpy.reshape(dates, (len(dates), 1))
	y = prices

	svr_lin = SVR(kernel='linear', C=1e3)
	svr_poly = SVR(kernel='poly', C=1e3, degree=2)
	svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

	svr_lin.fit(x, y)
	svr_poly.fit(x, y)
	svr_rbf.fit(x, y)

	plt.scatter(x, y, color='black', label='Data')
	plt.plot(x, svr_lin.predict(x), color='green', label='Linear model')
	plt.plot(x, svr_poly.predict(x), color='red', label='Polynomial model')
	plt.plot(x, svr_rbf.predict(x), color='blue', label='RBF model')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

	return svr_lin.predict(input)[0], svr_poly.predict(input)[0], svr_rbf.predict(input)[0]

load_data('pacb.csv')

prediction = predict_price(40)

print prediction
