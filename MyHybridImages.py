import numpy as np
from MyConvolution import convolve

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

    # Generate Gaussian Kernels
    low_pass_kernel = makeGaussianKernel(lowSigma)
    high_pass_kernel = makeGaussianKernel(highSigma)

    # Ensure images have the same shape
    if lowImage.shape != highImage.shape:
        raise ValueError("Input images must have the same shape.")

    # Apply low-pass filter to lowImage
    low_pass_filtered = convolve(lowImage, low_pass_kernel)

    # Apply high-pass filter to highImage
    high_pass_filtered = highImage - convolve(highImage, high_pass_kernel)

    # Create hybrid image by combining low-pass and high-pass images
    hybrid_image = low_pass_filtered + high_pass_filtered + 0.5  # Add 0.5 for visualization

    return hybrid_image

def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Create a 2D Gaussian kernel with standard deviation sigma.
    The kernel values sum to 1.0, and the size is floor(8*sigma+1) or floor(8*sigma+1)+1 (whichever is odd).
    :param sigma: standard deviation of the Gaussian kernel
    :type float
    :returns: 2D Gaussian kernel
    :rtype: np.ndarray
    """
    size = int(8.0 * sigma + 1.0)
    if size % 2 == 0:
        size += 1

    kernel = np.fromfunction(lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * np.exp(- ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)), (size, size))

    return kernel / np.sum(kernel)  # Normalize the kernel so that it sums to 1.0
