import csv
import cv2
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_string('folder', './', "The folder.")
lines = []
with open(FLAGS.folder + '/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = FLAGS.folder + '/IMG/' + filename
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
from keras.layers import Flatten, Dense, Convolution2D, Lambda, MaxPooling2D, Dropout, Conv2D

input_shape=(160,320,3)
num_classes=1
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
model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=(160,320,3)))

model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(num_classes))



model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=FLAGS.epochs)

model.save('bmodel.h5')
