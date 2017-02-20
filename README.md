# udacity.carnd.vehicle-detection-and-tracking

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The HOG features were extracted from my training images using the `get_hog_features` function in `main.py`. This function calls the `skimage.feature.hog` from the `skikit-image` module.

I've chosen to extract the HOG features from all three channels in an image and concatenate them into a single feature vector as this showed superior classification performance using an SVM classifier as discussed below.

The HOG space for a few sample images from the training set are displayed here:

*INSERT IMAGES*

####2. Explain how you settled on your final choice of HOG parameters.

The parameters for the HOG algorithm live in a larger `FeatureParameters` class defined in `utils.py`. I arrived at the values there by trial and error. The HOG operation is the most time consuming of the vehicle detection pipeline so it's advantageous to keep the number of features as small as possible for this reason. Also the pixel per cell size of (8,8) was chosen to be roughly the size of important features making up a car, i.e. the size of tail lights, windows, etc. In the interest of fewer features I've also reduced the orientation parameter down to 8 from the suggested value of 9 form the class notes with not noticeable degradation in classification accuracy.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
