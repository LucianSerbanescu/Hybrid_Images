# Hybrid_Images

# Overview
**A little program which uses a hand-coded convolution function to create hybrid images.**

*Hybrid Images* are static images that change in interpretation as a function of the viewing distance. The basic idea is that high frequency tends to dominate perception when it is available, but, at a distance, only the low frequency (smooth) part of the signal can be seen. By blending the high frequency portion of one image with the low-frequency portion of another, you get a hybrid image that leads to different interpretations at different distances. An example of a hybrid image is shown below.

![Screenshot 2023-10-24 at 1.31.46 pm.png](Hybrid%20Images%20Samples%2FScreenshot%202023-10-24%20at%201.31.46%E2%80%AFpm.png)

# Convolution function implementation
- the program is written in python
- uses a hand-coded convolution function which is not implemented using Fourier Trasnformation
- the convolution function uses arbitrary shaped kernels as long as both dimensions are odd (e.g. 7x9 kernels but not 4x5 kernels)
- the image has 0-padding, the conflict made with the kernel is handled by adding an extra layer of pixels around the image as a border
- both coloured and gray-scale images can be convolved
- the coloured image convolution was handled by iterating though each colour channel

# Hybrid Images filtering implementation
- hybrid image is the sum of low-pass filtered version of one image and high-pass filtered version of the second image
- there is a free parameter, which can be tuned for each image pair, which controls how much high frequency to remove from the first image and how much low frequency to leave in the second image. 
- this is called the "cutoff-frequency"
- the high-pass filtering it is achieved by subtracting a low-pass version of an image from itself
- 

## Gaussian Filtering
- low pass filtering (removing all the high frequencies) can be achieved by convolving the image with a Gaussian filter
- the cutoff-frequency is controlled by changing the standard deviation, sigma, of the Gaussian filter used in constructing the hybrid images.
- the code presents a function used to generate 2D Gaussian convolution kernel



# Example Image Set 

- there are a set of images in the folder "Hybrid Images Sample"
![Screenshot 2023-10-24 at 1.45.49 pm.png](Hybrid%20Images%20Samples%2FScreenshot%202023-10-24%20at%201.45.49%E2%80%AFpm.png)

- the low-pass (blurred) and high-pass versions of these images look like this:
![Screenshot 2023-10-24 at 1.45.54 pm.png](Hybrid%20Images%20Samples%2FScreenshot%202023-10-24%20at%201.45.54%E2%80%AFpm.png)

- result :
![Screenshot 2023-10-24 at 1.45.09 pm.png](Hybrid%20Images%20Samples%2FScreenshot%202023-10-24%20at%201.45.09%E2%80%AFpm.png)