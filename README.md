# Hybrid_Images
Little program which uses convolutional function to create hybrid images

# Overview

**A little program which uses convolutional function to create hybrid images based on SIGGRAPH 2006 paper by Olivia, Torralba and Schyns.**

*Hybrid Images* are static images that change in interpretation as a function of the viewing distance. The basic idea is that high frequency tends to dominate perception when it is available, but, at a distance, only the low frequency (smooth) part of the signal can be seen. By blending the high frequency portion of one image with the low-frequency portion of another, you get a hybrid image that leads to different interpretations at different distances. An example of a hybrid image is shown below.

![Screenshot 2023-10-24 at 1.31.46 pm.png](Hybrid%20Images%20Samples%2FScreenshot%202023-10-24%20at%201.31.46%E2%80%AFpm.png)

# Details

- template convolution. Template convolution is a fundamental image processing tool.
- the program support arbitrary shaped kernels, as long as both dimensions are odd (e.g. 7x9 kernels but not 4x5 kernels). 
- also uses zero-padding of the input image to ensure that the output image retains the same size as the input image and that the kernel can reach into the image edges and corners. 
- the implementation also support convolution of both grey-scale and colour images. 
- note that colour convolution is achieved by applying the convolution operator to each of the colour bands separately (i.e. treating each band as an independent grey-level image).

# Hybrid Images Description

- a hybrid image is the sum of a low-pass filtered version of the one image and a high-pass filtered version of a second image. 
- there is a free parameter, which can be tuned for each image pair, which controls how much high frequency to remove from the first image and how much low frequency to leave in the second image. 
- this is called the "cutoff-frequency". 
- in the paper it is suggested to use two cutoff-frequencies (one tuned for each image) and you are free to try that, as well.
- low pass filtering (removing all the high frequencies) can be achieved by convolving the image with a Gaussian filter
- the cutoff-frequency is controlled by changing the standard deviation, sigma, of the Gaussian filter used in constructing the hybrid images.

# Image Set 

- there are a set of images in the folder "Hybrid Images Sample"
![Screenshot 2023-10-24 at 1.45.49 pm.png](Hybrid%20Images%20Samples%2FScreenshot%202023-10-24%20at%201.45.49%E2%80%AFpm.png)

- the low-pass (blurred) and high-pass versions of these images look like this:
![Screenshot 2023-10-24 at 1.45.54 pm.png](Hybrid%20Images%20Samples%2FScreenshot%202023-10-24%20at%201.45.54%E2%80%AFpm.png)

- result :
![Screenshot 2023-10-24 at 1.45.09 pm.png](Hybrid%20Images%20Samples%2FScreenshot%202023-10-24%20at%201.45.09%E2%80%AFpm.png)