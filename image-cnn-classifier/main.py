# Based on tutorial https://www.youtube.com/watch?v=cAICT4Al5Ow&index=11&list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import numpy
from keras import backend as K
K.set_image_dim_ordering('th')

img_size = 150
batch_size = 16

epochs = 200
steps_per_epoch = 256

datagen = ImageDataGenerator() #rescale=1./255)

training_generator = datagen.flow_from_directory('data/training', target_size=(img_size, img_size), batch_size=batch_size, class_mode='binary')

validation_generator = datagen.flow_from_directory('data/validation', target_size=(img_size, img_size), batch_size=batch_size * 2, class_mode='binary')

print 'Training classes: ' + str(training_generator.class_indices)
print 'Validation classes: ' + str(validation_generator.class_indices)

model = Sequential()

def conv_layer(first=False):
	if first:
		#model.add(ZeroPadding2D(padding=(1, 1)))
		model.add(Convolution2D(64, (3, 3), input_shape=(3, img_size, img_size), data_format='channels_first'))
	else:
		model.add(Convolution2D(64, (3, 3), data_format='channels_first'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))

conv_layer(first=True)
conv_layer()
conv_layer()

model.add(Flatten())
model.add(Dense(64, input_shape=(batch_size, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

checkpoint = ModelCheckpoint('models/cnn.{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit_generator(training_generator, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=validation_generator, validation_steps=32, callbacks=[checkpoint])

model.save_weights('models/cnn.h5')

img = image.load_image('data/validation/cats/981.jpg', target_size=(224, 224))
prediction = model.predict(img)
print prediction
