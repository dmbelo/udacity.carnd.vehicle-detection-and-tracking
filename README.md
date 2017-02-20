# udacity.carnd.vehicle-detection-and-tracking

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The HOG features were extracted from my training images using the `get_hog_features` function in `main.py`. This function calls the `skimage.feature.hog` from the `skikit-image` module.

I've chosen to extract the HOG features from all three channels in an image and concatenate them into a single feature vector as this showed superior classification performance using an SVM classifier as discussed below.

Here are 5 examples from each class of the training set:

![alt text][image1]

The HOG space for a a car example is displayed below. Note the the gradients vaguely look like the car. Also not that 3 different hog channel are displayed in the hog diagram with red, green and blue gradients for each channel.

![alt text][image2]

In contrast, the non-car HOG looks like this.

![alt text][image3]

####2. Explain how you settled on your final choice of HOG parameters.

The parameters for the HOG algorithm live in a larger `FeatureParameters` class defined in `utils.py`. I arrived at the values there by trial and error. The HOG operation is the most time consuming of the vehicle detection pipeline so it's advantageous to keep the number of features as small as possible for this reason. Also the pixel per cell size of (8,8) was chosen to be roughly the size of important features making up a car, i.e. the size of tail lights, windows, etc. In the interest of fewer features I've also reduced the orientation parameter down to 8 from the suggested value of 9 form the class notes with not noticeable degradation in classification accuracy.

The choice of color space was a very influential one. I experimented with many different color spaces and channels, but in the end, using all channels of the YCrCb color space seemed to yield the best performance.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear support vector machine (SVM) using `sklearn.svm.SVC`. The training process was carried out in the `train.py` file. Here I first instantiate a `FeatureParameters` object which contains all of my relevant parameters controlling the feature extraction process. I then read in and aggregate all of my training images for car examples and non-car examples. The data set consisted of 8792 car samples and 8968 non-car samples. Once aggregated I extracted the features relative them using the `extract_features` function and created a label vector representative of the car/notcar classes. The data was scaled using `sklearn.preprocessing.StandardScaler`, randomized and divided into training and testing sets. Finally the classifier was fit to the training data and the accuracy was observed to be > 0.99.

Finally, the SVM and the scaler object are packed into a `Classifier` object and pickled into a file to be saved to the hard-disk and reloaded for vehicle detection later.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search was implemented in the function `slide_and_serch` within the `utils.py` file and used in the `detect_vehicle.py` file in the `pipeline` function. This function was implemented used a few paremeters which are defined in the `SearchParameters` class. A single fixed rectangular search window was used and the vertical search domain of the image was limited to roughly the bottom half in order to improve the search speed. For speed, the HOG features were extracted for the entire image before the windowing was performed. In order to extract the appropriate section of the hog results when windowing, the original image is scaled before windowing in order to be consistent with the hog feature retrieval used for the training process on the 64x64 training images.

That scale and overlap of the windows are largely based on trial and error. The sliding window search is an extremely computationally expensive operation, so a minimal amount of windows is desired. A balance of enough resolution of vehicle detection and execution speed was behind the decision of overlap and scale.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I used a set of 6 test images from the video to assess the performance of my classifier and sliding window algorithm. To optimize the peformance of my classifier I iterated the training process of the classifier. Whereas the test set that was set aside from the training set was a reasonable measure of the accuracy of the classifier, these images are a more realistic measure of the false-positive that the classifier needs to minimize. As well, the parameters of the sliding window and heat map thresholding are tuned iteratively by analzying the performance on the following 6 test images:



### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach I took in this project was that of a traditional machine vision approach which can largely be split into two steps: 1) fitting a classifier and 2) using a sliding window search to detect vehicle using the classifier.

As far as fitting the classifier, I would like to look further into the dataset and see if there are any opportunities to improve the false-positive detection of the classifier. Perhaps there is a bias of the classifier that can be exposed and remedied by augmenting the data set with better examples of cars and note cars.

The sliding window search method that I employed has room for a lot of improvement. At the moment what I've done is very basic and I struggle to deal with false-positives. I can see various options to improve its performance significantly including:
* Averaging the most recent heatmaps in order to better reject false-positives across time
* Implement a sliding window approach that varies the size of the search windows depending on how far down the road you are searching. The idea is that the vehicle that are further down the road should be smaller in size due to the perspective and thus require smaller search windows.
* Implement a sliding window approach that limits the search domain in the x-dimension as you approach the horizon


[//]: # (Image References)
[image1]: examples/car_notcar.png
[image2]: examples/car_hog.png
[image3]: examples/notcar_hog.png
<!-- [image3]: ./examples/sliding_windows.jpg -->
<!-- [image4]: ./examples/sliding_window.jpg -->
<!-- [image5]: ./examples/bboxes_and_heat.png -->
<!-- [image6]: ./examples/labels_map.png -->
<!-- [image7]: ./examples/output_bboxes.png -->
<!-- [video1]: ./project_video.mp4 -->
