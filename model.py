import csv
import cv2
import numpy as np
import tensorflow as tf

# command line flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_string('folder', './', "The folder.")

# read driving log from sim
lines = []
with open(FLAGS.folder + '/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# load all images from sim and corresponding measurements. save in parallel lists
images = []
measurements = []
for line in lines:

	# center images
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = FLAGS.folder + '/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)

	# steering correction for side images
	correction = 0.2

	# left images, adjust steering back to center of road
	source_path = line[1]
	filename = source_path.split('/')[-1]
	current_path = FLAGS.folder + '/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	measurements.append(measurement+correction)

	# right images, adjust steering back to center of road
	source_path = line[2]
	filename = source_path.split('/')[-1]
	current_path = FLAGS.folder + '/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	measurements.append(measurement-correction)

# flip images and corresponding measurements to generate twice as much traning data
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(measurement*-1.0)

# store data in numpy array
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# import libraries from keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Lambda, MaxPooling2D, Cropping2D

# static definitions
input_shape=(160,320,3)
num_classes=1

# setup Keras model, normalize, and crop
model = Sequential()
model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=input_shape))
model.add(Cropping2D(cropping=((70,25),(0,0))))

# LeNet model
# model.add(Convolution2D(6,5,5,activation='relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6,5, 5, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(num_classes))

# Nvidia model
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(num_classes))

# compile model, train, and save the training history
model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=FLAGS.epochs)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
savefig('model.png')
