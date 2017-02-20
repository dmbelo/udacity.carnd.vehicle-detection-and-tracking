import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

from scipy.ndimage.measurements import label as scipy_label
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.externals import joblib



class Classifier():
    def __init__(self, clf, scaler):
        self.clf = clf
        self.scaler = scaler

    def predict(self, features):
        """
        Wrapper of SVM.predict which includes feature scaling
        """
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        return self.clf.predict(features_scaled)


class FeatureParameters():
    def __init__(self):
        self.spatial_size = (16, 16)
        self.hist_bins = 32
        self.orientations = 8
        self.pixels_per_cell = 10
        self.cells_per_block = 2


class SearchParameters():
    def __init__(self):
        self.window_size = 64
        self.scale = 1.4
        self.cell_per_step = 2
        self.ystart = 380
        self.ystop = 660


def read_image(file):
    img = cv2.imread(file)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def process_image(img):
    img = img.astype(np.float32)/255
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)


def print_stats(cars, notcars):
    """
    Function to print some characteristics of the dataset
    """
    print("Number of car samples: {0}".format(len(cars)))
    print("Number of non car samples: {0}".format(len(notcars)))
    img = cv2.imread(cars[0])
    print("Image shape: {0}x{1}".format(img.shape[0], img.shape[1]))
    print("Image datatype: {}".format(img.dtype))


def get_hog_features(img, p, visualize=False, feature_vector=False):

    channels = range(img.shape[2])
    features = []
    img_hog = np.zeros_like(img)

    for i, channel in enumerate(channels):
        out = hog(img[:, :, channel], orientations=p.orientations,
                  pixels_per_cell=(p.pixels_per_cell, p.pixels_per_cell),
                  cells_per_block=(p.cells_per_block, p.cells_per_block),
                  transform_sqrt=True, visualise=visualize,
                  feature_vector=feature_vector)
        if visualize:
            features.append(out[0])
            img_hog[:, :, i] = out[1]
        else:
            features.append(out)

    return features, img_hog


def get_color_hist_features(img, p):
    """
    Function to compute color histogram features
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=p.hist_bins)
    channel2_hist = np.histogram(img[:, :, 1], bins=p.hist_bins)
    channel3_hist = np.histogram(img[:, :, 2], bins=p.hist_bins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0],
                                    channel2_hist[0],
                                    channel3_hist[0]))
    return hist_features


def get_bin_color_features(img, p):
    """
    Function to compute binned color features
    """
    return cv2.resize(img, p.spatial_size).ravel()


def extract_features(img, p):
    bin_features = get_bin_color_features(img, p)
    hist_features = get_color_hist_features(img, p)
    hog_features = get_hog_features(img, p, feature_vector=True)[0]
    features = np.concatenate((bin_features, hist_features,
                               np.ravel(hog_features)))
    return features


def extract_window_features(img_draw, img_window, block_per_window,
                            hog_features, ypos, xpos, xleft, ytop, heatmap,
                            p_search, p_features, classifier):
    hog_feat0 = hog_features[0][ypos:ypos + block_per_window, xpos:xpos +
                                block_per_window].ravel()
    hog_feat1 = hog_features[1][ypos:ypos + block_per_window, xpos:xpos +
                                block_per_window].ravel()
    hog_feat2 = hog_features[2][ypos:ypos + block_per_window, xpos:xpos +
                                block_per_window].ravel()

    hog_features = np.hstack((hog_feat0, hog_feat1, hog_feat2))
    bin_features = get_bin_color_features(img_window, p_features)
    hist_features = get_color_hist_features(img_window, p_features)

    features = np.hstack((bin_features, hist_features, hog_features))
    prediction = classifier.predict(features.reshape(1, -1))

    if prediction == 1:
        xbox_left = np.int(xleft * p_search.scale)
        ytop_draw = np.int(ytop * p_search.scale)
        win_draw = np.int(p_search.window_size * p_search.scale)
        box = [(xbox_left, ytop_draw + p_search.ystart), (xbox_left + win_draw,
               ytop_draw + win_draw + p_search.ystart)]
        cv2.rectangle(img_draw, box[0], box[1],
                      (0, 0, 255), 6)
        add_heat(heatmap, box)


def add_heat(heatmap, box):
    # for box in bbox_list:
    heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def train(cars, notcars, p):
    features_car = []
    for car in cars:
        img = read_image(car)
        img_processed = process_image(img)
        features_car.append(extract_features(img_processed, p))

    features_notcar = []
    for notcar in notcars:
        img = read_image(notcar)
        img_processed = process_image(img)  # png
        features_notcar.append(extract_features(img_processed, p))

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
    clf = SVC(kernel="linear")
    # Check the training time for the SVC
    t = time.time()
    clf.fit(features_train, labels_train)
    print('{0:2.2f} seconds to train SVC...'.format(time.time() - t))

    # Accuracy score
    accuracy = clf.score(features_test, labels_test)
    print('Test Accuracy of SVC = {0:2.4f}'.format(accuracy))

    return Classifier(clf, scaler)


def slide_and_search(img_search, img_draw, hog_features,
                     classifier, p_search, p_features):
    heatmap = np.zeros_like(img_draw[:, :, 0], np.dtype(np.float32))

    nxblocks = np.int(img_search.shape[1] / p_features.pixels_per_cell) - 1
    nyblocks = np.int(img_search.shape[0] / p_features.pixels_per_cell) - 1
    block_per_window = p_search.window_size // p_features.pixels_per_cell - 1

    nxsteps = np.int((nxblocks - block_per_window) / p_search.cell_per_step)
    nysteps = np.int((nyblocks - block_per_window) / p_search.cell_per_step)

    for xstep in range(nxsteps):
        for ystep in range(nysteps):
            ypos = ystep * p_search.cell_per_step
            xpos = xstep * p_search.cell_per_step
            xleft = xpos * p_features.pixels_per_cell
            ytop = ypos * p_features.pixels_per_cell
            windowed_image = cv2.resize(img_search[ytop:ytop +
                                        p_search.window_size, xleft:xleft +
                                        p_search.window_size], (64, 64))

            extract_window_features(img_draw, windowed_image,
                                    block_per_window, hog_features, ypos, xpos,
                                    xleft, ytop, heatmap, p_search, p_features,
                                    classifier)

    return heatmap


def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


def draw_car_boxes(img, boxes):
    # Iterate through all detected cars
    bboxes = []
    for car_number in range(1, boxes[1]+1):
        # Find pixels with each car_number label value
        nonzero = (boxes[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return bboxes


def pipeline(img, classifier, p_features, p_search):
    img_draw_search = img.copy()
    img_draw_cars = img.copy()
    img_processed = process_image(img)
    img_search = img_processed[p_search.ystart:p_search.ystop, :, :]
    shape = img_search.shape
    img_search = cv2.resize(img_search, (np.int(shape[1] / p_search.scale),
                                         np.int(shape[0] / p_search.scale)))
    hog_features = get_hog_features(img_search, p_features)[0]
    heatmap = slide_and_search(img_search, img_draw_search, hog_features,
                               classifier, p_search, p_features)

    heatmap_thresh = heatmap.copy()
    apply_threshold(heatmap_thresh, 2)
    boxes = scipy_label(heatmap_thresh)

    draw_car_boxes(img_draw_cars, boxes)

    return img_draw_search, img_draw_cars, heatmap


if __name__ == "__main__":
    p_features = FeatureParameters()
    p_search = SearchParameters()
    #
    # import glob
    # notcars = glob.glob('data/non-vehicles/*/*.png')
    # cars = glob.glob('data/vehicles/*/*.png')
    #
    # print_stats(cars, notcars)
    # classifier = train(cars, notcars, p_features)
    # joblib.dump(classifier, 'classifier.pkl')
    #
    # # Time to predict
    # img_test = read_image('data/vehicles/GTI_Far/image0018.png')
    # img_processed = process_image(img_test)
    # features_test = extract_features(img_processed, p_features)
    # t = time.time()
    # prediction = classifier.predict(features_test.reshape(1, -1))
    # print('{0:2.2f} seconds to make prediction...'.format(time.time() - t))
    # print(prediction)

    classifier = joblib.load('classifier.pkl')

    img = read_image('test_images/test6.jpg')
    t = time.time()
    img_draw_search, img_draw_cars, heatmap = pipeline(img, classifier,
                                                       p_features, p_search)

    print('{0:2.2f} seconds to run pipeline...'.format(time.time() - t))

    plt.subplot(121)
    plt.imshow(img_draw_cars)
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')

    plt.show()
