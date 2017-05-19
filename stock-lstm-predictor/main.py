import csv
import numpy
from os import listdir
from os.path import isfile, join
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, TimeDistributed
from keras.utils import np_utils, generic_utils
import matplotlib.pyplot as plt

max_features = 10000
maxlen = None
timesteps = 10
pred_length = 5
batch_size = 10
epochs = 2

def plot(prices, volumes, predictions):
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	ax1.scatter(range(len(prices)), prices, color='black', label='Price')
	ax1.scatter(range(len(volumes)), volumes, color='blue', label='Volume')
	ax1.plot(range(len(predictions)), predictions, color='red', label='Prediction')
	
	plt.xlabel('Day')
	plt.ylabel('Value')
	plt.title('LSTM Prediction')
	plt.legend()
	plt.show()

model = Sequential()

x_train = []
y_train = []

def load_data(dir):
	global x_train
	global y_train
	global maxlen

	data_files = [file for file in listdir(dir) if isfile(join(dir, file))]
	
	x = []

	min0 = 99999999999
	max0 = 0
	min1 = 99999999999
	max1 = 0

	for data_file in data_files:
		with open(join(dir, data_file), 'r') as file:
			data_points = []

			data_reader = csv.reader(file)
			next(data_reader)
			for row in data_reader:
				#outputs.append(float(row[1]))
				if float(row[1]) < min0:
					min0 = float(row[1])
				if float(row[1]) > max0:
					max0 = float(row[1])
				if float(row[5]) < min1:
					min1 = float(row[5])
				if float(row[5]) > max1:
					max1 = float(row[5])
				
				data_points.append([float(row[1]), float(row[5])])

			# Do min/max scaling to 0-1
			for data_point in data_points:
				data_point[0] = (float(data_point[0]) - min0) / (max0 - min0)
				data_point[1] = (float(data_point[1]) - min1) / (max1 - min1)

			x.append(data_points)
	
	if maxlen == None:
		x = sequence.pad_sequences(x, dtype=float)
		maxlen = 0
		
		for seq in x:
			if len(seq) > maxlen:
				maxlen = len(seq)

	else:
		x = sequence.pad_sequences(x, maxlen=maxlen, dtype=float)

	x_train_tmp = numpy.array(x, dtype=numpy.float)
	y_train = [] # numpy.array(y)
	
	flag = 0
	
	#print x_train_tmp.shape
	
	for sample in range(x_train_tmp.shape[0]):
		x_tmp = numpy.array([x_train_tmp[sample, i:i+(timesteps),:] for i in range(x_train_tmp.shape[1] - (timesteps + pred_length))])
		
		y_tmp = numpy.array([x_train_tmp[sample, i + timesteps + pred_length,1] for i in range(x_train_tmp.shape[1] - (timesteps + pred_length))])
		
		if flag == 0:
			x_train = x_tmp
			y_train = y_tmp
			
		else:
			x_train = numpy.concatenate((x_train, x_tmp))
			y_train = numpy.concatenate((y_train, y_tmp))

	#y_train = [np_utils.to_categorical(x) for x in y_train]

	y_train = numpy.reshape(y_train, (maxlen - timesteps - pred_length, 1))

	print x_train.shape
	print y_train.shape
	
	return

def train_model():
	#model.add(Embedding(max_features, 32, input_shape=x_train.shape))
	model.add(TimeDistributed(Dense(maxlen - timesteps - pred_length, activation='softmax', input_shape=x_train.shape[1:]), input_shape=x_train.shape[1:]))
	model.add(LSTM(maxlen - timesteps - pred_length, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

	score, accuracy = model.evaluate(x_train, y_train, batch_size=batch_size)

	print('Score: ' + str(score))
	print('Accuracy: ' + str(accuracy))

	plot_prices = []
	plot_volumes = []
	plot_predictions = []
	
	for data_set in x_train:
		for data_point in data_set:
			plot_prices.append(data_point[0])
			plot_volumes.append(data_point[1])
		plot_predictions.append(model.predict(data_set))
	
	plot(plot_prices, plot_volumes, plot_predictions)

	return score, accuracy

load_data('data')
train_model()
