import cv2
import numpy as np
from MyHybridImages import myHybridImages
# import cProfile
# import sample
# cProfile.run('sample')
import time

# Load low and high frequency images
low_freq_image_1 = cv2.imread("data/dog.bmp")
high_freq_image_1 = cv2.imread("data/cat.bmp")

low_freq_image_2 = cv2.imread("data/einstein.bmp")
high_freq_image_2 = cv2.imread("data/marilyn.bmp")

low_freq_image_3 = cv2.imread("data/bicycle.bmp")
high_freq_image_3 = cv2.imread("data/motorcycle.bmp")

low_freq_image_4 = cv2.imread("data/bird.bmp")
high_freq_image_4 = cv2.imread("data/plane.bmp")

low_freq_image_5 = cv2.imread("data/fish.bmp")
high_freq_image_5 = cv2.imread("data/submarine.bmp")

# high_freq_personal_image = cv2.imread("personal_example/dali_painting.jpg")
# low_freq_personal_image = cv2.imread("personal_example/abraham_lincon.jpg")

# high_freq_personal_image = cv2.imread("personal_example/costume.jpg")
# low_freq_personal_image = cv2.imread("personal_example/icecream.jpg")


# Convert the image to NumPy array
# low_freq_image = np.array("data/dog.bmp")
# high_freq_image = np.array("data/cat.bmp")

# Set the standard deviations for low-pass and high-pass filtering
# low_sigma = 4
# high_sigma = 3

# for the personal image
# lincon
# low_sigma = 10
# high_sigma = 20

low_sigma = 5
high_sigma = 10

# Create hybrid image
hybrid_image_1 = myHybridImages(low_freq_image_1, low_sigma, high_freq_image_1, high_sigma)
# hybrid_image_2 = myHybridImages(low_freq_image_2, low_sigma, high_freq_image_2, high_sigma)
# hybrid_image_3 = myHybridImages(low_freq_image_3, low_sigma, high_freq_image_3, high_sigma)
# hybrid_image_4 = myHybridImages(low_freq_image_4, low_sigma, high_freq_image_4, high_sigma)
# hybrid_image_5 = myHybridImages(low_freq_image_5, low_sigma, high_freq_image_5, high_sigma)
# personal_example = myHybridImages(low_freq_personal_image, low_sigma, high_freq_personal_image, high_sigma)

# Display or save the hybrid image as needed

cv2.imwrite("hybridImage_1.jpg", hybrid_image_1)
# cv2.imwrite("output_images/hybridImage_2.jpg", hybrid_image_2)
# cv2.imwrite("output_images/hybridImage_3.jpg", hybrid_image_3)
# cv2.imwrite("output_images/hybridImage_4.jpg", hybrid_image_4)
# cv2.imwrite("output_images/hybridImage_5.jpg", hybrid_image_5)
# cv2.imwrite("output_images/personal_example.jpg", personal_example)




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
