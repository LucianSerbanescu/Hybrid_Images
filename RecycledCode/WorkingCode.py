import cv2
import numpy as np


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image_height, image_width, colour_channels = image.shape if len(image.shape) == 3 else (
    image.shape[0], image.shape[1], 1)
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]  # Convert grayscale image to single channel

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
    result = np.zeros(image.shape, dtype=float)

    for c in range(colour_channels):
        for y in range(image_height):
            for x in range(image_width):
                roi = padded_image[y:y + kernel_height, x:x + kernel_width, c]
                conv_value = np.sum(roi * kernel)
                result[y, x, c] = conv_value

    return result


def apply_high_pass_filter(image, high_freq_sigma=30):
    # Apply custom convolution to the image with the high-pass filter kernel
    high_pass_kernel = np.array([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])
    high_freq_output = convolve(image, high_pass_kernel)

    # Normalize the output image
    high_freq_output = (high_freq_output - np.min(high_freq_output)) * 255 / (
                np.max(high_freq_output) - np.min(high_freq_output))
    high_freq_output = high_freq_output.astype(np.uint8)

    return high_freq_output


def apply_low_pass_filter(image, blur_kernel_size=15):
    # Apply Gaussian blur to the image to keep only the low frequencies
    low_pass_image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)

    return low_pass_image


def create_hybrid_image(high_freq_image_path, low_freq_image_path, high_freq_radius=30, blur_kernel_size=15):
    # Load the colored images
    high_freq_image = cv2.imread(high_freq_image_path)
    low_freq_image = cv2.imread(low_freq_image_path)

    # Apply high-pass filter to the high-frequency image
    high_freq_output = apply_high_pass_filter(high_freq_image, high_freq_radius)

    # Apply low-pass filter to the low-frequency image
    low_freq_output = apply_low_pass_filter(low_freq_image, blur_kernel_size)

    # Create the hybrid image by combining high and low frequencies
    hybrid_image = cv2.addWeighted(low_freq_output, 0.4, high_freq_output, 0.9, 0)

    return high_freq_output, low_freq_output, hybrid_image


# Example usage:
high_freq_image_path = '../data/cat.bmp'  # Replace with the actual path to your high-frequency image
low_freq_image_path = '../data/dog.bmp'  # Replace with the actual path to your low-frequency image
high_freq_output, low_freq_output, hybrid_image = create_hybrid_image(high_freq_image_path, low_freq_image_path)

# Display the high-frequency image
cv2.imshow('High Frequency Image', high_freq_output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the low-pass filtered image using Gaussian distribution
cv2.imshow('Low Pass Filtered Image', low_freq_output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the hybrid image
cv2.imshow('Hybrid Image', hybrid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
