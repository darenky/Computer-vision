import os
import numpy as np
from PIL import Image

def custom_erosion(image, kernel):
    img_array = np.array(image)
    height, width = img_array.shape
    k_height, k_width = kernel.shape
    result = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if img_array[i:i+k_height, j:j+k_width].min() == 255:
                result[i, j] = 255

    result_image = Image.fromarray(result, mode='L')

    return result_image

def custom_dilation(image, kernel):
    img_array = np.array(image)
    height, width = img_array.shape
    k_height, k_width = kernel.shape
    result = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if img_array[i:i+k_height, j:j+k_width].max() == 255:
                result[i:i+k_height, j:j+k_width] = 255

    result_image = Image.fromarray(result, mode='L')

    return result_image

def custom_closing(image, kernel):
    dilated_image = custom_dilation(image, kernel)
    closed_image = custom_erosion(dilated_image, kernel)

    return closed_image

def custom_opening(image, kernel):
    eroded_image = custom_erosion(image, kernel)
    opened_image = custom_dilation(eroded_image, kernel)

    return opened_image

structuring_element = np.array([[1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1]], dtype=np.uint8)

input_image = Image.open("Example.png").convert("L")  

output_directory = "output"
os.makedirs(output_directory, exist_ok=True)

erosion_result = custom_erosion(input_image, structuring_element)
erosion_result.save(os.path.join(output_directory, "Eroded.png"))

dilation_result = custom_dilation(input_image, structuring_element)
dilation_result.save(os.path.join(output_directory, "Dilated.png"))

opening_result = custom_opening(input_image, structuring_element)
opening_result.save(os.path.join(output_directory, "Opened.png"))

closing_result = custom_closing(input_image, structuring_element)
closing_result.save(os.path.join(output_directory, "Closed.png"))

