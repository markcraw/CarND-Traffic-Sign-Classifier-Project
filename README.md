#**Traffic Sign Recognition** 

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

[image1]: ./data_set_stats.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./9.jpg "Traffic Sign 1"
[image5]: ./13.jpg "Traffic Sign 2"
[image6]: ./34.jpg "Traffic Sign 3"
[image7]: ./35.jpg "Traffic Sign 4"
[image8]: ./38.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the types and amounts data used in tthis project.  It shows values for the training, validation and testing data.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it speeded up the training and reduced the overall size of the model. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it helps to fairly treat the various features that the model will learn.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale imag   						| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x46x24 				|
| Drop out  	      	| 40% keep rate                 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x12  8 				|
| Drop out  	      	| 40% keep rate                 				|
| Fully connected		| inputs 499, outputs 120        				|
| RELU					|												|
| Drop out  	      	| 40% keep rate                 				|
| Softmax				| cross_entropy									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters:
- learning rate = 0.001
- optimizer = AdamOptimizer
- batch size = 128
- epochs = 15

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I tried several architectures where I modified the filter size and the the number of filters. THe accuracy of the model incresed slightly, but I was not able to acheive the 93% accuracy on the validation set.

* What were some problems with the initial architecture?
The model under fit the data and it did not achieve a very high accuracy on the test data.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I modifed the number of filters for the CNN layes and added dropout try to reduce the overfitting of the model

* Which parameters were tuned? How were they adjusted and why?
I adjusted the parameters based on trial and error to improve the accuracy.

If a well known architecture was chosen:
* What architecture was chosen?
I started with the LeNet architecture.

* Why did you believe it would be relevant to the traffic sign application?
It displayed good performanceon other visual applications.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because it is very similar to the "no entry" sign, which has a center feature in the circle. The second imagemay be easier to classify because it is the only image which has an inverted triangle as a central feature.  The third image may be difficult to identify becuase it is a turn left sign which is similar to the turn right sign.  The fourth image may be difficult to classify becuase it is similar to the go strait or lef/rightt signs .The fifth sign may be difficult to classify because is similar to the keep left sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction            					| 
|:---------------------:|:---------------------------------------------:| 
| No overtaking      	| No entry  									| 
| Give way     			| Give way 										|
| Turn left ahead		| Turn left ahead								|
| Ahead only	    	| Ahead only					 				|
| Keep right		    | Keep right     		    					|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of validation and test datasets.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located below the cell "Predict the Sign Type for Each Image" of the Ipython notebook.

For the first image, the model is relatively sure that this is a no entry (probability of 0.135), and the image does not contain a no entryn. The top five soft max probabilities were

| Probability         	|     Prediction(class number) 					| 
|:---------------------:|:---------------------------------------------:| 
| .135         			| 17 						            		| 
| .114     				| 9	    								    	|
| .038					| 14				                			|
| .035	      			| 34					 			        	|
| .027				    | 38      			    	        			|


For the second image, the model is relatively sure that this is a give way (probability of 0.395), and the image does contain a give way. The top five soft max probabilities were

| Probability         	|     Prediction(class number) 					| 
|:---------------------:|:---------------------------------------------:| 
| .395         			| 13 						            		| 
| .075     				| 15									    	|
| .062					| 9					                			|
| .046	      			| 12					 			        	|
| .019				    | 38      			    	        			|

For the third image, the model is relatively sure that this is turn left ahead (probability of 0.088), and the image does contain a turn left ahead. The top five soft max probabilities were

| Probability         	|     Prediction(class number) 					| 
|:---------------------:|:---------------------------------------------:| 
| .088         			| 34 						            		| 
| .031     				| 15									    	|
| .029					| 35				                			|
| .022	      			| 9 					 			        	|
| .007				    | 37      			    	        			|

For the fourth image, the model is relatively sure that this is a ahead only (probability of 0.151), and the image does contain an ahead only. The top five soft max probabilities were

| Probability         	|     Prediction(class number) 					| 
|:---------------------:|:---------------------------------------------:| 
| .151         			| 35 						            		| 
| .058     				| 37									    	|
| .033					| 18				                			|
| .030	      			| 38					 			        	|
| .026				    | 3      			    	        			|

For the fifth image, the model is relatively sure that this is a keep right (probability of 0.054), and the image does contain a keep right. The top five soft max probabilities were

| Probability         	|     Prediction(class number) 					| 
|:---------------------:|:---------------------------------------------:| 
| .054         			| 38 						            		| 
| .051     				| 19									    	|
| .028					| 23				                			|
| .004	      			| 10					 			        	|
| -.004				    | 29      			    	        			|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


