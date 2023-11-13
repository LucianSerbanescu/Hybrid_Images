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

    # print(lowImage)

    # Low-pass filter lowImage
    low_kernel = makeGaussianKernel(lowSigma)
    low_pass_filtered = convolve(lowImage, low_kernel)

    # Print the entire array
    # print(my_array)
    #     print(row)

    # print(low_kernel)
    # print(low_pass_filtered)
    # cv2.imwrite("low_pass_filtered.jpg", low_pass_filtered)

    # Normalize the pixel values to the range [0, 1]
    # low_pass_filtered_normalised = low_pass_filtered / 255.0
    # cv2.imshow('low_pass_filtered', low_pass_filtered_normalised)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(highImage)

    # High-pass filter highImage
    high_kernel = makeGaussianKernel(highSigma)
    high_pass_filtered = highImage - convolve(highImage, high_kernel)

    # print(high_kernel)
    # print(high_pass_filtered)
    # cv2.imwrite("high_pass_filtered.jpg", high_pass_filtered)

    # # Normalize the pixel values to the range [0, 1]
    # high_pass_filtered_normalised = high_pass_filtered / 255.0 + 0.5
    # cv2.imshow('high_pass_filtered', high_pass_filtered_normalised)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite("high_pass_filtered_normalised.jpg", high_pass_filtered_normalised)

    # Combine low-pass and high-pass images to create the hybrid image
    hybrid_image = low_pass_filtered + high_pass_filtered
    # print(hybrid_image)
    cv2.imwrite("hybrid_image.jpg", hybrid_image)

    # Normalize the pixel values to the range [0, 1] for using cv2.show
    # The image with contrast 1:1 with the example
    hybrid_image_normalised = hybrid_image / 255.0
    cv2.imshow('hybrid_image_normalised', hybrid_image_normalised)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
        size = size + 1  # Ensure the size is odd

    # Calculate the center of the kernel
    center = size // 2

    # Create a grid of coordinates from -center to center
    x, y = np.mgrid[-center:center + 1, -center:center + 1]

    # Calculate the Gaussian values using the 2D formula
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # Normalize the kernel so that the sum of values is 1.0
    kernel /= kernel.sum()

    return kernel
