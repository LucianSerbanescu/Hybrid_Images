import numpy as np

def pad_image(image, kernel_size):
    pad_width = kernel_size // 2
    return np.pad(image, pad_width, mode='constant', constant_values=0)

def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve an image with a kernel assuming zero-padding of the image to handle the borders
    :param image: the image (shape=(rows,cols) or shape=(rows,cols,channels)) :type numpy.ndarray
    :param kernel: the kernel (odd by odd shape=(kheight,kwidth)) :type numpy.ndarray
    :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray
    """
    kheight, kwidth = kernel.shape
    if kheight % 2 == 0 or kwidth % 2 == 0:
        raise ValueError("Kernel dimensions must be odd by odd.")

    if len(image.shape) == 2:
        image = image.reshape(image.shape[0], image.shape[1], 1)  # Convert grayscale to color image

    pad_image_array = pad_image(image, kwidth)
    rows, cols, channels = image.shape
    output_image = np.zeros((rows, cols, channels), dtype=float)

    for i in range(kheight):
        for j in range(kwidth):
            # Perform element-wise multiplication between sub-image and kernel for each channel
            output_image[:, :, :] += pad_image_array[i:i + rows, j:j + cols, :] * kernel[i, j]

    return output_image + 0.5  # Add 0.5 to make negative values positive in visualization
