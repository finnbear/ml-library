# Based on tutorial: https://www.youtube.com/watch?v=vOppzHpvTiQ

import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt

dataframe = pandas.read_fwf('brain-body.txt')

x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

body_reg = linear_model.LinearRegression()

body_reg.fit(x_values, y_values)

plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
