# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia_cnn_image]: ./images/cnn-architecture-624x890.png "Nvidia Model Visualization"
[center]: ./images/center_2016_12_01_13_33_08_951.jpg "Center Driving"
[left]: ./images/recovery_left.png "Left Recovery Image"
[right]: ./images/recovery_right.png "Right Recovery Image"
[normal]: ./images/center_2016_12_01_13_31_13_279.jpg "Normal Image"
[flipped]: ./images/center_2016_12_01_13_31_13_279_flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_hilnbrand.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based of the Nvidia self-driving car model and consists of a convolution neural network with both 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 88-92) 

The model includes RELU layers to introduce nonlinearity (code lines 88-92), the data is normalized in the model using a Keras lambda layer (code line 74), and is cropped using Keras cropping layer, removing the top 70 pixels and the bottom 25 pixels (code line 75). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 102). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 101).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, as well as driving the track in the opposite direction. When I recorded training data, however, I used the mouse, which drove the car very smoothly, but only generated angles which seemed to be between -1 and 1. When I trained the model on this data, the steering angles used to autonomously drive the car seemed to be within the same range, which was not enough to go around the tight turns. I tried using the provided data.zip data, which produced much higher turning angles, most likely because it was created using the WASD keys, which is either on at 25/-25 degrees, or off at 0 degrees. This resulted in much more responsive driving after training the model.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a proven model, as recommended in the course videos. I used the Nvidia self-driving car model.

My first step was to use a series of convolution neural network models. This is appropriate because it processes the road road images to classify major objects which can identify a left or right turning road, such as curvature of the left lane line.

Then I connected flattened the data to pass it through four additional dense layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. about 20% of the training set is used for validation.

To combat the overfitting, I only ran two epochs. More then two epochs displayed an oscillating behavior of the rms error in the validation set, as well as an eventual increase in rms error in the training model. Two epochs seems to work well.

The final step was to run the simulator to see how well the car was driving around track one. There was one spot where the vehicles pulled to the left and crossed over the yellow line, but did not touch the curb, and then eventually corrected itself as it went around a gradual left curve.

When I trained my model using by data, and started using the left and right images to train corrections, I realized that the network worked mediocre to go around left turns but barely turned the wheels to go around right. I was sure that I trained it to handle both left and right corrections, but upon further review of my model, I realized that I had copy the code to process the left camera data and pasted to process the right camera data, and changed the measurements to subract the correction to turn left rather than right like the other camera data would, but I forgot to ge the right camera data. This, actually, processed the left camera images again, training to also turn to the left when its at the left, making for weird behavior. I fixed the code to extract the right camera data as well, and it worked much better. I did end up using the provided training data in the end though.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 74-98) consisted of a convolution neural network with the following layers and layer sizes:

* Normalize, between -0.5 and 0.5
* Cropping, 70 pixels off the top and 25 pixels off the bottom
* Apply a 5x5 convolution with 24 output filters, 2x2 subsample, RELU activation
* Apply a 5x5 convolution with 36 output filters, 2x2 subsample, RELU activation
* Apply a 5x5 convolution with 48 output filters, 2x2 subsample, RELU activation
* Apply a 3x3 convolution with 64 output filters, RELU activation
* Apply a 3x3 convolution with 64 output filters, RELU activation
* Flatten
* Dense, output size 100
* Dense, output size 50
* Dense, output size 10
* Dense, output size 1

Here is a visualization of the architecture:

![alt text][nvidia_cnn_image]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back towards the center when it finds itself off-course. I recorded two examples on each side. These images show what a recovery looks like starting from the left and the right :

![alt text][left]
![alt text][right]

I also drove the track in the reverse direction two or three times around.

To augment the data sat, I also flipped images and angles thinking that this would double the amoung of data to train on, as they generate the same scenario but turning the opposite way. This will provide a perfect balance of left and right turning training. For example, here is an image that has then been flipped:

![alt text][normal]
![alt text][flipped]

After the collection process, I had 48,216 number of data points. I then preprocessed this data by normalizing the image and cropping the top and bottom, as described earlier.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by the training and evaluation rms error, as described above. I used an adam optimizer so that manually training the learning rate wasn't necessary.
