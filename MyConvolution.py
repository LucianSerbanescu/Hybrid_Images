import numpy as np

def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve an image with a kernel assuming zero-padding of the image to handle the borders
    :param image: the image (shape=(rows, cols, channels))
    :type numpy.ndarray

    :param kernel: the kernel (odd by odd shape=(kheight, kwidth))
    :type numpy.ndarray

    :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray
    """
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Kernel dimensions must be odd numbers")

    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')

    output = np.zeros_like(image, dtype=np.float32)  # Initialize output image with float32 type

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                output[i, j, k] = np.sum(padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1], k] * kernel)

    return output
