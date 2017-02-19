import glob
import cv2
import time
import numpy as np

from utils import Features
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split


def print_stats(cars, notcars):
    """
    Function to print some characteristics of the dataset
    """
    print("Number of car samples: {0}".format(len(cars)))
    print("Number of non car samples: {0}".format(len(notcars)))

    img = cv2.imread(cars[0])
    print("Image shape: {0}x{1}".format(img.shape[0], img.shape[1]))

    print("Image datatype: {}".format(img.dtype))


def process_image(file):
    img = cv2.imread(file)
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)


def train(cars, notcars):
    # Initialize feature object
    feat_obj = Features(spatial_size=(32, 32),
                        hist_bins=32,
                        orientations=9,
                        pixels_per_cell=20,
                        cells_per_block=2)

    features_car = []
    for car in cars:
        img = process_image(car)
        features_car.append(feat_obj.extract(img))

    features_notcar = []
    for notcar in notcars:
        img = process_image(notcar)
        features_notcar.append(feat_obj.extract(img))

    features = np.vstack((features_car, features_notcar)).astype(np.float64)
    # Fit a per-column scaler
    scaler = StandardScaler().fit(features)
    # Apply the scaler to X
    features_scaled = scaler.transform(features)
    # Define the labels vector
    labels = np.hstack((np.ones(len(features_car)),
                        np.zeros(len(features_notcar))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    out = train_test_split(features_scaled, labels, test_size=0.2,
                           random_state=rand_state)
    features_train, features_test, labels_train, labels_test = out

    # Initialize support vector machine object
    clf = SVC(kernel="linear")
    # Check the training time for the SVC
    t = time.time()
    clf.fit(features_train, labels_train)
    print('{0:2.2f} seconds to train SVC...'.format(time.time() - t))

    # Accuracy score
    accuracy = clf.score(features_test, labels_test)
    print('Test Accuracy of SVC = {0:2.4f}'.format(accuracy))


if __name__ == "__main__":
    # List of data set image files
    notcars = glob.glob('data/non-vehicles_smallset/*/*')
    cars = glob.glob('data/vehicles_smallset/*/*')

    print_stats(cars, notcars)
    train(cars, notcars)
