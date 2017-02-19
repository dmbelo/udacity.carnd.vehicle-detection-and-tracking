import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.feature import hog


class Features():
    def __init__(self, img, spatial_size, hist_bins):
        self.img = img
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.hog_features = get_hog_features(img)

    def extract(self):
        spatial_features = get_bin_spatial_features(self.img, size=self.spatial_size)
        hist_features = get_color_hist_features(self.img, nbins=self.hist_bins)
        features = np.concatenate((spatial_features,
                                   hist_features,
                                   self.hog_features))
        return features

    def extract_from_tile(self):
        raise NotImplementedError


def get_hog_features(img, orientations, pixels_per_cell, cells_per_block,
                     visualise=False, feature_vector=True):

    if visualise:
        out = hog(img, orientations=orientations,
                  pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                  cells_per_block=(cells_per_block, cells_per_block),
                  transform_sqrt=True, visualise=visualise,
                  feature_vector=feature_vector)
        return out[0], out[1]  # features, hog_image
    else:
        features = []
        for channel in range(img.shape[2]):
            out = hog(img[:, :, channel], orientations=orientations,
                      pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                      cells_per_block=(cells_per_block, cells_per_block),
                      transform_sqrt=True, visualise=visualise,
                      feature_vector=feature_vector)
            features.append(out)
        return np.ravel(features)


def get_bin_spatial_features(img, size=(32, 32)):
    """
    Function to compute binned color features
    """
    return cv2.resize(img, size).ravel()


def get_color_hist_features(img, nbins=32):
    """
    Function to compute color histogram features
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0],
                                    channel2_hist[0],
                                    channel3_hist[0]))
    return hist_features


def sliding_window_search(img,
                          x_start_stop=[None, None],
                          y_start_stop=[None, None],
                          xy_window=(64, 64),
                          xy_overlap=(0.5, 0.5)):

    # If x and/or y start/stop positions not defined, set to image size
    # Compute the span of the region to be searched
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    x_span = x_start_stop[1] - x_start_stop[0]
    y_span = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows_ = x_span//nx_pix_per_step - 1
    ny_windows_ = y_span//ny_pix_per_step - 1
    # Deal with odd cases
    nx_windows = nx_windows_ + np.int((x_span - nx_windows_) >= xy_window[0])
    ny_windows = ny_windows_ + np.int((y_span - ny_windows_) >= xy_window[1])

    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window positions
            x_start = xs * nx_pix_per_step + x_start_stop[0]
            x_end = x_start + xy_window[0]
            y_start = ys*ny_pix_per_step + y_start_stop[0]
            y_end = y_start + xy_window[1]
            window_list.append(((x_start, y_start), (x_end, y_end)))

    return window_list


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    img_copy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(img_copy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return img_copy


if __name__ == "__main__":
    img_file = '000528.png'
    img = cv2.imread(img_file)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(image.shape)
    windows = sliding_window_search(image,
                                    x_start_stop=[100, 399],
                                    y_start_stop=[150, 249],
                                    xy_window=(200, 200),
                                    xy_overlap=(0.0, 0.0))

    window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)

    plt.imshow(window_img)
    plt.show()
