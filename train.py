import numpy as np
import glob
import time

from utils import Classifier, FeatureParameters, extract_features, print_stats
from utils import read_image, process_image
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


parameters = FeatureParameters()


def train():
    notcars = glob.glob('data/non-vehicles/*/*.png')
    cars = glob.glob('data/vehicles/*/*.png')

    print_stats(cars, notcars)

    features_car = []
    for car in cars:
        img = read_image(car)
        img_processed = process_image(img)
        features_car.append(extract_features(img_processed, parameters))

    features_notcar = []
    for notcar in notcars:
        img = read_image(notcar)
        img_processed = process_image(img)  # png
        features_notcar.append(extract_features(img_processed, parameters))

    features = np.vstack((features_car, features_notcar))
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
    clf = SVC(kernel='linear', C=0.00001)
    # Check the training time for the SVC
    t = time.time()
    clf.fit(features_train, labels_train)
    print('{0:2.2f} seconds to train SVC...'.format(time.time() - t))

    # Accuracy score
    accuracy = clf.score(features_test, labels_test)
    print('Test Accuracy of SVC = {0:2.4f}'.format(accuracy))

    classifier = Classifier(clf, scaler)
    joblib.dump(classifier, 'classifier.pkl')

    return classifier


if __name__ == '__main__':
    train()
