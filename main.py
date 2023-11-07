import cv2
import numpy as np
from MyHybridImages import myHybridImages

# Load low and high frequency images
low_freq_image = cv2.imread("data/dog.bmp")
high_freq_image = cv2.imread("data/cat.bmp")

# Define standard deviations for low-pass and high-pass filters
low_sigma = 10
high_sigma = 5

# Ensure images are in the same shape (you might need to resize them if necessary)
# low_freq_image = cv2.resize(low_freq_image, (high_freq_image.shape[1], high_freq_image.shape[0]))

# Create hybrid image
hybrid_image = myHybridImages(low_freq_image, low_sigma, high_freq_image, high_sigma)

# Clip the resulting image to ensure valid pixel values
hybrid_image = np.clip(hybrid_image, 0, 255).astype(np.uint8)

# Display or save the hybrid image as needed
cv2.imshow("Hybrid Image", hybrid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
