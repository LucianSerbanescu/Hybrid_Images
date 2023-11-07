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
    # Your code here. You'll need to vectorise your implementation to ensure it runs # at a reasonable speed.


