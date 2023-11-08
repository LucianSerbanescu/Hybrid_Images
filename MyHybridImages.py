import numpy as np
from MyConvolution import convolve
from math import floor
import cv2


def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma: float) -> np.ndarray:
    """
    Create hybrid images by combining a low-pass and high-pass filtered pair.
    :param lowImage: the image to low-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray
    :param lowSigma: the standard deviation of the Gaussian used for low-pass filtering lowImage
    :type float
    :param highImage: the image to high-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray
    :param highSigma: the standard deviation of the Gaussian used for low-pass filtering highImage before subtraction to create the high-pass filtered image
    :type float
    :returns returns the hybrid image created
        by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it with
        a high-pass image created by subtracting highImage from highImage convolved with
        a Gaussian of s.d. highSigma. The resultant image has the same size as the input images.
    :rtype numpy.ndarray
    """

    # Low-pass filter lowImage
    low_kernel = makeGaussianKernel(lowSigma)
    low_pass_filtered = convolve(lowImage, low_kernel)
    cv2.imwrite("low_pass_filtered.jpg", low_pass_filtered)

    # High-pass filter highImage
    high_kernel = makeGaussianKernel(highSigma)
    high_pass_filtered = highImage - convolve(highImage, high_kernel)

    # Adjust the pixel values of the high-pass filtered image for visualization
    high_pass_visualized = high_pass_filtered + 0.5
    high_pass_visualized = ((high_pass_visualized - (-127.5)) / (127.5 - (-127.5))) * 255
    high_pass_visualized = np.clip(high_pass_visualized, 0, 255).astype(np.uint8)

    cv2.imwrite("high_pass_filtered.jpg", high_pass_visualized)

    # Combine low-pass and high-pass images to create the hybrid image
    hybrid_image = low_pass_filtered + high_pass_filtered

    return hybrid_image

def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Use this function to create a 2D gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or
    floor(8*sigma+1)+1 (whichever is odd) as per the assignment specification.
    """
    # Calculate the size of the kernel
    size = int(floor(8 * sigma + 1))
    if size % 2 == 0:
        size += 1  # Ensure the size is odd

    # Calculate the center of the kernel
    center = size // 2

    # Create a grid of coordinates from -center to center
    x, y = np.mgrid[-center:center + 1, -center:center + 1]

    # Calculate the Gaussian values using the 2D formula
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # Normalize the kernel so that the sum of values is 1.0
    kernel /= kernel.sum()

    return kernel
