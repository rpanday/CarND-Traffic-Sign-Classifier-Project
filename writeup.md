# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/explore-viz.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./images/20.png "20"
[image4]: ./images/60.png "60"
[image5]: ./images/ahead.png "Ahead"
[image6]: ./images/ahead-left.png "Ahead left"
[image7]: ./images/cycling.png "Cycling"
[image8]: ./images/caution.png "Caution"
[image9]: ./images/speed_bump.png "Speed Bump"
[image10]: ./examples/top5-predictions.png "Predictions"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/rpanday/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set in code cell **[3]** of notebook.

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Here I have listed down the various classes with their label, description and one sample image from training set. This can be found in code cell **[6]**

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because when I visualized the images provided they had variable brightness, exposure and shadowing. Converting to grayscale made the features of traffic signs stand out and also helps training to run faster. This is done in code cell **[7]**.

As a last step, I normalized the image data because I wanted to have zero mean and equal variance in the data. Normalization didn't affect the image output ofcourse but helped the model get stable gradients, reach minima faster and not worry about scale differences of different dimensions. This is done in code cell **[9]**.

Here is an example of a traffic sign images after grayscaling & normalization. The traffic signs can be seen better now by just eyeballing.

![alt text][image2]

I did not generate additional data because by experimenting with hyperparameters I managed to get the validation accuracy of above 93%.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is the original LENET5 with different output size. LENET5 had the output size of 10 but in my model the output size is 43. This is done in code cell **[12]**.
It consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten               | 1D array with 400 elements
| Fully connected	    | 120 outputs     									|
| Fully connected	    | 84 outputs     									|
| Fully connected (Logits)	    | 43 outputs     									|
|                       |              |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I followed the approach from lessons leading up to this project. I used Adam optimizer which performs better than stochastic gradient descent. After experimenting, I finally settled with batch size of 128, number of epochs was 100, learning rate was 0.001. I also used a mu of 0 and sigma of 0.1 for generating random weights and biases while training our model. This is done in code cell **[13], [15]**.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 95.7% 
* test set accuracy of 93.4%
Related code cells **[18], [19]**

I chose a well known architecture.
* LENET5 architecture was chosen with different output size
* LENET paper linked in the lectures describes that model was used for document recogniztion and here the problem is traffic sign recognition. Both problems rely on identifying shapes and patterns so I thought it is relevant.
* The accuracy is 100% on training set but it also 95.7% on validation and 93.4% on test sets. So there is no problem of overfitting or underfitting here but with good training.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I found 7 images from the web that I used for prediction. These images are not exactly German traffic signs and also images are picked from stock image sites with watermarks on them. So a considerable noise does exist in these images. This can be seen in code cell **[20]**
Here are the images:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image3]

I think the image for Ahead left, Speed Bump and Cycling may be difficult to classify while the speed limit signs should be easier. But not all of my images are from German Traffic signs to it would be interesting to see the top predictions and how close they are.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction. This can be found in code cell **[22]**:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 35. Ahead      		| 35. Ahead   									| 
| 34. Ahead left     			| 40. Roundabout 										|
| 3. 60 Km/h					| 2. 50 Km/h											|
| 29. Cycling	      		| 29. Cycling					 				|
| 18. Caution			| 18. Caution      							|
| 22. Speed Bump    | 22. Speed Bump |
| 0. 20 Km/h        | 3. 60 Km/h    |
|   |   |


The model was able to correctly guess 4 of the 7 traffic signs, which gives an accuracy of 57%. This is in contrast to the 95.7% accuracy of the test set. The surprising part is that images like speed limits 20 Km/h and 60 Km/h were not predicted correctly but the complicated shapes for cycling, speed bump were idetified and model was able to differentiate between caution and ahead signs too though both look close. 
While 20 Km/h and 60 Km/h were not identified correctly the model gives close looking suggestion as a matter of fact in 50 Km/h and 60 Km/h respectively.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code showing top 5 guesses is located in cell **[23]** of the Ipython notebook.

The top five soft max probabilities were

![alt text][image10]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
I did not attempt this section.

