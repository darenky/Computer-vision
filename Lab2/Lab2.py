import os
import numpy as np
from PIL import Image

image_path = "Example.jpg"
image = np.array(Image.open(image_path).convert("L"))

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Kernels 
invertion = np.array([[-1, -1, -1],
                      [-1, 8, -1],
                      [-1, -1, -1]])

sharpening = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

my_filter = np.array([[-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, 49, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]])

sobel_horizontal_operator = np.array([[-1, 0, 1],
                                      [-2, 0, 2],
                                      [-1, 0, 1]])

sobel_vertical_operator = np.array([[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]])

# Gaussian filter (11 x 11): 
def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2) / (2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

gaussian_kernel_size = 11
gaussian_sigma = 1.5
gaussian_kernel_11x11 = gaussian_kernel(gaussian_kernel_size, gaussian_sigma)

# Diagonal Blurring (7 x 7):
diagonal_blur_kernel_size = 7
diagonal_blur_kernel = np.eye(diagonal_blur_kernel_size, dtype=float)
diagonal_blur_kernel /= np.sum(diagonal_blur_kernel)

def edge_detection_filter(image):
    height, width = image.shape
    edge_result = np.zeros_like(image, dtype=float)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            horizontal_gradient = np.sum(image[i - 1:i + 2, j - 1:j + 2] * sobel_horizontal_operator)
            vertical_gradient = np.sum(image[i - 1:i + 2, j - 1:j + 2] * sobel_vertical_operator)

            edge_result[i, j] = np.sqrt(horizontal_gradient**2 + vertical_gradient**2)

    edge_result = np.clip(edge_result, 0, 255)
    return edge_result.astype(np.uint8)

# Move image 20down and 10right
moved_image = np.zeros((41, 41))
moved_image[0, 9] = 1

def apply_convolution(image, kernel):
    image_height, image_width = image.shape
    kernel_size = kernel.shape[0]
    padding = kernel_size // 2
    result = np.zeros((image_height - 2 * padding, image_width - 2 * padding))

    for i in range(padding, image_height - padding):
        for j in range(padding, image_width - padding):
            result[i - padding, j - padding] = np.sum(image[i - padding:i + padding + 1, j - padding:j + padding + 1] * kernel)

    return result.clip(0, 255).astype(np.uint8)

def save_result(output_dir, result, filename):
    Image.fromarray(result).save(os.path.join(output_dir, filename))

result_filters = {
    'Moved.jpg': moved_image,
    'Inverted.jpg': invertion,
    'Gaussian_blurred.jpg': gaussian_kernel_11x11,
    'Diagonal_blurred.jpg': diagonal_blur_kernel,
    'Sharpened.jpg': sharpening,
    'Sobel.jpg': sobel_vertical_operator,
    'My_filter.jpg': my_filter,
    'Edge_detected.jpg': sobel_horizontal_operator # Horizontal edges will be more visible than vertical
}

for filename, kernel in result_filters.items():
    result = apply_convolution(image, kernel)
    save_result(output_dir, result, filename)


