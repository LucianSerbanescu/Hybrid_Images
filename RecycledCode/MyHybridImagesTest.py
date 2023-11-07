import numpy as np
import cv2
from MyConvolutionTest import convolve

def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma: float, cutoffFrequencyLow: float, cutoffFrequencyHigh: float) -> np.ndarray:
    """
    Create hybrid images by combining a low-pass and high-pass filtered pair.
    :param lowImage: the image to low-pass filter (either grayscale shape=(rows,cols) or color shape=(rows,cols,channels))
    :type numpy.ndarray
    :param lowSigma: the standard deviation of the Gaussian used for low-pass filtering lowImage
    :type float
    :param highImage: the image to high-pass filter (either grayscale shape=(rows,cols) or color shape=(rows,cols,channels))
    :type numpy.ndarray
    :param highSigma: the standard deviation of the Gaussian used for low-pass filtering highImage before subtraction to create the high-pass filtered image
    :type float
    :param cutoffFrequencyLow: cutoff frequency for lowImage (controls how much high frequency to remove)
    :type float
    :param cutoffFrequencyHigh: cutoff frequency for highImage (controls how much low frequency to leave)
    :type float
    :returns returns the hybrid image created
        by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it with
        a high-pass image created by subtracting highImage from highImage convolved with
        a Gaussian of s.d. highSigma. The resultant image has the same size as the input images.
    :rtype numpy.ndarray
    """

    # Create low-pass Gaussian kernels with specified cutoff frequencies
    low_kernel = makeGaussianKernel(lowSigma, cutoffFrequencyLow)
    high_kernel = makeGaussianKernel(highSigma, cutoffFrequencyHigh)

    # Low-pass filter the lowImage
    low_pass = convolve(lowImage, low_kernel)

    # High-pass filter the highImage
    high_pass = highImage - convolve(highImage, high_kernel)

    # Create the hybrid image by adding low-pass and high-pass components
    hybrid_image = low_pass + high_pass

    # Add 0.5 to make negative values positive in visualization
    hybrid_image += 0.5
    #
    # # Display the hybrid image on the screen
    # cv2.imshow('Hybrid Image', hybrid_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # # If you want to save the hybrid image
    # cv2.imwrite('hybrid_image.jpg', (hybrid_image * 255).astype(np.uint8))

    return hybrid_image

def makeGaussianKernel(sigma: float, size: int, channels: int = 1) -> np.ndarray:
    """
    Use this function to create a 2D or 3D Gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0, and the size should be an odd number.
    """
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )

    # For 3D kernel (color images), replicate the 2D kernel across channels
    if channels > 1:
        kernel = np.stack([kernel] * channels, axis=-1)

    # Normalize the kernel to make sure it sums up to 1
    kernel /= np.sum(kernel)

    return kernel

