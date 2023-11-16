import numpy as np


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve an image with a kernel assuming zero-padding of the image to handle the borders
    :param image: the image (shape=(rows,cols) or shape=(rows,cols,channels))
    :type numpy.ndarray

    :param kernel: the kernel (odd by odd shape=(kheight,kwidth))
    :type numpy.ndarray

    :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray
    """

    # set channels depending on if it is coloured image or grayscale image
    if len(image.shape) == 3:
        image_height, image_width, colour_channels = image.shape
    elif len(image.shape) == 2:
        image_height, image_width = image.shape
        colour_channels = 1
    else:
        raise ValueError("Input image must be 2D (grayscale) or 3D (color)")

    # set up the kernel
    kernel_height, kernel_width = kernel.shape

    # check if the kernel is odd
    if kernel_height % 2 == 0 or kernel_width % 2 == 0:
        raise ValueError("Kernel dimensions must be odd")

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # put the paddings to the image
    padded_image = pad_image(image, pad_height, pad_width)

    # this initialization step is necessary because the convolve function calculates the convolution results and
    # stores them in the output array. By initializing output with zeros, the function ensures that it starts with a
    # clean slate before accumulating the convolution results.
    output = np.zeros_like(image, dtype=np.float32)

    # the actual convolution with vectorization
    for i in range(image_height):
        for j in range(image_width):
            for channel in range(colour_channels):
                output[i, j, channel] = np.sum(
                    padded_image[i: (i + kernel_height), j: (j + kernel_width), channel] * kernel)

    return output


def pad_image(image: np.ndarray, pad_height: int, pad_width: int) -> np.ndarray:
    """
    Pad the input image with zeros.
    :param image: the image (shape=(rows,cols) or shape=(rows,cols,channels))
    :type numpy.ndarray

    :param pad_height: number of rows to pad on top and bottom
    :type int

    :param pad_width: number of columns to pad on left and right
    :type int

    :returns the padded image
    :rtype numpy.ndarray
    """
    padding = ((pad_height, pad_height), (pad_width, pad_width))

    # If the image is a color image, add padding for channels
    if len(image.shape) == 3:
        padding = padding + ((0, 0),)

    return np.pad(image, padding, mode='constant')
