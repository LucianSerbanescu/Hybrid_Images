import cv2
import numpy as np

def hybrid_images_color(image1, image2, cutoff_frequency=10):
    # Split input images into color channels
    b1, g1, r1 = cv2.split(image1)
    b2, g2, r2 = cv2.split(image2)

    # Apply Gaussian filter to each color channel to get low-frequency components
    low_freq_b1 = cv2.GaussianBlur(b1.astype(np.float32), (2*cutoff_frequency+1, 2*cutoff_frequency+1), 0)
    low_freq_g1 = cv2.GaussianBlur(g1.astype(np.float32), (2*cutoff_frequency+1, 2*cutoff_frequency+1), 0)
    low_freq_r1 = cv2.GaussianBlur(r1.astype(np.float32), (2*cutoff_frequency+1, 2*cutoff_frequency+1), 0)

    low_freq_b2 = cv2.GaussianBlur(b2.astype(np.float32), (2*cutoff_frequency+1, 2*cutoff_frequency+1), 0)
    low_freq_g2 = cv2.GaussianBlur(g2.astype(np.float32), (2*cutoff_frequency+1, 2*cutoff_frequency+1), 0)
    low_freq_r2 = cv2.GaussianBlur(r2.astype(np.float32), (2*cutoff_frequency+1, 2*cutoff_frequency+1), 0)

    # Subtract low-frequency components to get high-frequency components
    high_freq_b1 = b1 - low_freq_b1
    high_freq_g1 = g1 - low_freq_g1
    high_freq_r1 = r1 - low_freq_r1

    high_freq_b2 = b2 - low_freq_b2
    high_freq_g2 = g2 - low_freq_g2
    high_freq_r2 = r2 - low_freq_r2

    # Combine low-frequency component of image1 with high-frequency component of image2
    hybrid_b = low_freq_b1 + high_freq_b2
    hybrid_g = low_freq_g1 + high_freq_g2
    hybrid_r = low_freq_r1 + high_freq_r2

    # Merge color channels to obtain the colored hybrid image
    hybrid_image = cv2.merge((hybrid_b, hybrid_g, hybrid_r))

    # Convert the hybrid image to uint8 type
    hybrid_image = hybrid_image.astype(np.uint8)

    return hybrid_image

# Load images
image1 = cv2.imread('data/cat.bmp')
image2 = cv2.imread('data/dog.bmp')

# Set cutoff frequency for Gaussian filter
cutoff_frequency = 10

# Generate colored hybrid image
colored_hybrid_image = hybrid_images_color(image1, image2, cutoff_frequency)

# Display colored hybrid image
cv2.imshow('Colored Hybrid Image', colored_hybrid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
