# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nVidia_model.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py: containing the script to create and train the model
* drive.py: for driving the car in autonomous mode
* model.h5: containing a trained convolution neural network 
* writeup_report.md: summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I have tried several models:

* Simple fully connected neural network: Input -> Dense(100) -> Dense(50) -> Dense(1). However, the performance is very bad in the mean-squared error sense. Consequently, it is required to consider adding more layers to complexify the model.
* Simple fully connected neural network + convolution2d layers: Input -> Conv2D with 5 x 5 filters with stride at 2, with 16 filters-> maxpooling layer -> Dense(100) -> Dense(1).
* Subsequently, I have used the NVidia network model.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. In particular, we use 25 percent of the data for validation and the remainder of them for training. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The data provided by Udacity, consisting of 8036 entries. For each row of data sample, it consists of following information:
* Center image: The image captured by camera positioned at the center of the car
* Left image: The image captured by camera positioned on the left side of the car
* Right image: The image captured by camera positioned on the right side of the car
* The angle information
* The throttle information
* The speed information
For the purpose of this project, only the first four entries at each row are used. Training data was chosen to keep the vehicle driving on the road. For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I have conducted the research on this project following a few steps: 
* Data pre-processing: (i) the original image has size 160 x 320 x 3, with entries from 0 - 255, in this case the MSE is very large and the convergence speed is very slow. Thus, we first normalize the data to the interval of -0.5 to 0.5. (ii) secondly, we notice that a few information in the image is useless, for example, most of the top part of the image consists of tree sky and surrounding environment, which are irrelavant to the prediction. Moreover, they are noise to the neural network. As a consequence, we cropped the image using the cropping layer in Keras. (iii) If we train the model using only the center image, the car may not know what to do when it hits the side of the road. To deal with this problem, we consider adding left and right images as features. In this case, the angle measurements of these images are unspecified. To address this issue, we use a hyperparameter - correction to add onto the existing measurement of the entry. More specifically, if a car is at the left side of the road, we would like it to move back middle, consequently, we add the measurement angle by the value correction, we do the opposite for the case when the car is on the right side of the road. (iv) In track 1, the car mostly turns left due to track construction. To deal with this issue, we consider flipping the images and negate the corresponding angle measurements to create more variety in the data.
* Model architecture design: We use the model proposed by NVidia autonomous car group with additional layer at the end, see the next subsection for more details.
* Model parameter tuning: There are a few parameters to be tuned: (i) the number of epochs, (ii) training validation split ratio, (iii)

Notice that without the data pre-processing step, the the vehicle fell off the track easily. In particular, when we dont have additional images capturing left or right side of the view, no matter how well the neural network is trained, the car fell off the road easily. With the introduction of these images, the performance of the car is more robust.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 70-84) consisted of a neural network model with the following layers:
Here is a visualization of the architecture

![alt text][image1]

#### 3. Creation of the Training Set & Training Process
After the collection process, I had 25715 number of data points. The data are shuffled randomly when training. I finally randomly shuffled the data set and put 25% of the data into a validation set. I used an adam optimizer so that manually training the learning rate wasn't necessary. 

The training and validation loss from epoch 1 to 5 is as follows:
Training MSE: 0.0182 -> 0.0152 -> 0.0143 -> 0.0134 -> 0.012
Validation MSE: 0.0172 -> 0.0148 -> 0.0145 -> 0.014 -> 0.0124