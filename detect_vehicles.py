import numpy as np
import cv2
import matplotlib.pyplot as plt
import os.path

from scipy.ndimage.measurements import label as scipy_label
from sklearn.externals import joblib
from moviepy.editor import VideoFileClip
from utils import Classifier, FeatureParameters, SearchParameters
from utils import get_hog_features, slide_and_search, apply_threshold
from utils import draw_car_boxes, process_image, read_image


p_features = FeatureParameters()
p_search = SearchParameters()
classifier = joblib.load('classifier.pkl')


def pipeline(img):
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
    apply_threshold(heatmap_thresh, 4)
    boxes = scipy_label(heatmap_thresh)

    draw_car_boxes(img_draw_cars, boxes)

    return img_draw_search, img_draw_cars, heatmap


def process_frame(img):
    out = pipeline(img)
    img_draw_search, img_draw_cars, heatmap = out
    return img_draw_cars


def detect_vehicles():
    file = 'project_video.mp4'
    # file = 'short_project_video.mp4'
    # file = 'test_images/test6.jpg'
    video_extensions = {'.mp4', '.mov'}
    extension = os.path.splitext(file)[1]
    if extension in video_extensions:
        video_output = "project_video_output.mp4"
        clip = VideoFileClip(file)
        clip = clip.fl_image(process_frame)
        clip.write_videofile(video_output, audio=False)
    else:
        img = read_image(file)
        out = pipeline(img)
        img_draw_search, img_draw_cars, heatmap = out
        plt.figure(figsize=(10, 3))
        plt.subplot(131)
        plt.imshow(img_draw_search)
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(heatmap, cmap='hot')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(img_draw_cars)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    detect_vehicles()
