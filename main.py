import cv2
import numpy as np
from MyHybridImages import myHybridImages
from MyConvolution import convolve

# Load low and high frequency images
low_freq_image = cv2.imread("data/dog.bmp")
high_freq_image = cv2.imread("data/cat.bmp")

# Set the standard deviations for low-pass and high-pass filtering
low_sigma = 4
high_sigma = 3

# Create hybrid image
hybrid_image = myHybridImages(low_freq_image, low_sigma, high_freq_image, high_sigma)

# Display or save the hybrid image as needed
cv2.imwrite("hybridImage.jpg", hybrid_image)



# # TESTING CONVOLUTION
#
# # Example kernel for testing
# kernel = np.array([[-1, -2, -1],
#                    [0, 0, 0],
#                    [1, 2, 1]])
#
# # Perform convolution on the color image with the kernel
# convolved_color_image = convolve(high_freq_image, kernel)
#
# cv2.imwrite("convolveFunction.jpg", convolved_color_image)
