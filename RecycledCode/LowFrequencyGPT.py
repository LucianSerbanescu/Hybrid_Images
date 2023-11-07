import cv2
import numpy as np


def apply_high_pass_filter(image, high_freq_radius=30):
    # Apply Fourier transform to the image
    f = np.fft.fft2(image, axes=(0, 1))
    fshift = np.fft.fftshift(f, axes=(0, 1))

    # Create a mask to filter out low frequencies (keep high frequencies)
    rows, cols = image.shape[:2]  # Get dimensions of the image (ignores the third dimension if present)
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = high_freq_radius  # Radius of the mask for high frequencies (adjust this value to control the filter)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0

    # Apply the mask to the shifted Fourier transform
    fshift = fshift * mask

    # Apply inverse Fourier transform to get the image with high frequencies
    f_ishift = np.fft.ifftshift(fshift, axes=(0, 1))
    img_back = np.fft.ifft2(f_ishift, axes=(0, 1))
    img_back = np.abs(img_back)

    # Convert the output image to uint8 type
    high_freq_output = np.uint8(img_back)

    return high_freq_output


def apply_low_pass_filter(image, blur_kernel_size=15):
    # Apply Gaussian blur to the image to keep only the low frequencies
    low_pass_image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)

    return low_pass_image


def create_hybrid_image(high_freq_image_path, low_freq_image_path, high_freq_radius=30, blur_kernel_size=15):
    # Load the colored images
    high_freq_image = cv2.imread(high_freq_image_path)
    low_freq_image = cv2.imread(low_freq_image_path)

    # Apply high-pass filter to the high-frequency image channels
    high_freq_output = np.zeros_like(high_freq_image)
    for i in range(3):  # Iterate over RGB channels
        high_freq_output[:, :, i] = apply_high_pass_filter(high_freq_image[:, :, i], high_freq_radius)

    # Apply low-pass filter to the low-frequency image channels
    low_freq_output = np.zeros_like(low_freq_image)
    for i in range(3):  # Iterate over RGB channels
        low_freq_output[:, :, i] = apply_low_pass_filter(low_freq_image[:, :, i], blur_kernel_size)

    # Create the hybrid image by combining high and low frequencies
    hybrid_image = cv2.addWeighted(low_freq_output, 0.5, high_freq_output, 0.5, 0)

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
