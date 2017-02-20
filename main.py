import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import os.path
from moviepy.editor import VideoFileClip

from scipy.ndimage.measurements import label as scipy_label
from skimage.feature import hog




p_features = FeatureParameters()
p_search = SearchParameters()

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


def process_frame(img):
    out = pipeline(img, classifier, p_features, p_search)
    img_draw_search, img_draw_cars, heatmap = out
    return img_draw_cars


def main(file):
    classifier = joblib.load('classifier.pkl')
    p_features = FeatureParameters()
    p_search = SearchParameters()

    video_extensions = {'.mp4', '.mov'}
    extension = os.path.splitext(file)[1]
    if extension in video_extensions:
        video_output = "project_video_output.mp4"
        clip = VideoFileClip("short_project_video.mp4")
        clip = clip.fl_image(process_frame)
        clip.write_videofile(video_output, audio=False)
    else:
        img = read_image(file)
        out = pipeline(img, classifier, p_features, p_search)
        img_draw_search, img_draw_cars, heatmap = out
        plt.subplot(121)
        plt.imshow(img_draw_search)
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.show()


if __name__ == "__main__":
    file = 'short_project_video.mp4'
    # file = 'test_images/test4.jpg'
    main(file)
