import os
import cv2
import numpy as np

input_dir = "photos"
output_dir = "output"

# Function to convert a color image to grayscale with gamma correction
def convert_to_grayscale(input_image, gamma=1.04):
    r_const, g_const, b_const = 0.2126, 0.7152, 0.0722
    grayscale_image = r_const * input_image[:, :, 2] ** gamma + g_const * input_image[:, :, 1] ** gamma + b_const * input_image[:, :, 0] ** gamma
    return grayscale_image.astype(np.uint8)

# Function to compute Otsu's criteria for a given threshold
def _compute_otsu_criteria(im, th):
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    if weight1 == 0 or weight0 == 0:
        return np.inf

    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]

    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

    return weight0 * var0 + weight1 * var1

# Function to binarize an image using Otsu's method
def otsuThresholding(img: np.ndarray) -> np.ndarray:
    threshold_range = range(np.max(img)+1)
    criterias = np.array([_compute_otsu_criteria(img, th) for th in threshold_range])

    best_threshold = threshold_range[np.argmin(criterias)]

    binary = img
    binary[binary > best_threshold] = 255
    binary[binary <= best_threshold] = 0

    return binary


image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

for image_file in image_files:
    input_image = cv2.imread(os.path.join(input_dir, image_file))
    grayscale_image = convert_to_grayscale(input_image)
    thresholded_image = otsuThresholding(grayscale_image)
    object_image = cv2.bitwise_and(input_image, input_image, mask=thresholded_image)
    object_path = os.path.join(output_dir, image_file)
    cv2.imwrite(object_path, object_image)