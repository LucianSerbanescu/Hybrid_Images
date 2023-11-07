import numpy as np
import cv2

def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image_height, image_width, colour_channels = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
    result = np.zeros(image.shape, dtype=float)

    for c in range(colour_channels):
        for y in range(image_height):
            for x in range(image_width):
                roi = padded_image[y:y + kernel_height, x:x + kernel_width, c]
                conv_value = np.sum(roi * kernel)
                result[y, x, c] = conv_value

    return result

image = cv2.imread('data/cat.bmp')
kernel = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

output = convolve(image, kernel)

output = (output - np.min(output)) * 255 / (np.max(output) - np.min(output))

output = output.astype(np.uint8)

cv2.imwrite('output.jpg', output)