import numpy as np
import matplotlib.pyplot as plt
import cv2


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
