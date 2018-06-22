import csv
import cv2
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 50, "The number of epochs.")

lines = []
with open('./driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = './IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Lambda

# model = Sequential()
# # add a 3x3 convolution on top, with 32 output filters:
# # model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(160,320,3)))
# # model.add(Convolution2D(32, 3, 3))
# model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=(160,320,3)))
# model.add(Convolution2D(6, 5, 5, activation='relu'))
# model.add(Flatten())
# # model.add(Flatten(input_shape=(160,320,3)))
# # model.add(Dense(100))
# model.add(Dense(1))

model = Sequential()
# For an explanation on conv layers see http://cs231n.github.io/convolutional-networks/#conv
# By default the stride/subsample is 1 and there is no zero-padding.
# If you want zero-padding add a ZeroPadding layer or, if stride is 1 use border_mode="same"
model.add(Convolution2D(12, 5, 5, activation = 'relu', input_shape=(160,320,3), init='he_normal'))

# For an explanation on pooling layers see http://cs231n.github.io/convolutional-networks/#pool
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(25, 5, 5, activation = 'relu', init='he_normal'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the 3D output to 1D tensor for a fully connected layer to accept the input
model.add(Flatten())
model.add(Dense(180, activation = 'relu', init='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(100, activation = 'relu', init='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'softmax', init='he_normal')) #Last layer with one output per class



model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=FLAGS.epochs)

model.save('bmodel.h5')
