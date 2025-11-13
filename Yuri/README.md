# GAN_II
The reconstruction of images on optical wavelengths with high precision has always been challenging for astronomers. Here, we present the application of a  Generative Adversarial Network (GAN) for Intensity Interferometry and perform image reconstruction for a fast-rotator.

The Python files follow the information of the model of the star and baseline configuration as follows:

## To create N number of Images....

1. ellipse.py: Define a function to create N number of 2-dimensional (x, y) grids to draw a contour map of an ellipse with different parameters (radius, inclination, position angle, axis ratio)
2. generate_image.py: The parameters of the ellipse are defined here and saved in .jpg format

## To simulate the Telescope and Baselines Configuration.......

1. aperture.py: It defines the shape of the aperture for each telescope and visualizes it on the observational plane. It also defines a function to create an array of baseline' names and length on  
                observational plane.
2. obstime.py: It calculates the number of steps at each observational night due to observational intervals. It also calculates the start and end time after each observational interval. There is 
               another definition to calculate an array of all Julian days for all given observational intervals.
3. obspoint.py: Since everything is calculated in radians. There are definitions to convert degrees and hours to radians. There is also another function that calculates the change in baseline due to 
                earth rotation
4. generate_baseline.py: It generates the variational baseline with time, which is used to simulate the II signal for any source and telescope array. However, we need to define some parameters here, 
                         exa, (1) Array of starting times of the observational day. (2) Array of ending times for the observational day (3) calculate the number of steps for the given observational interval 
                         (average time), so the array of Julian time (4) Define the telescope's position and name. Then get the baseline's name and length. (5) Define the position of of source in the sky 
                         coordinate. (6) Calculate the variational baseline according to the Earth's rotation. (7) Plot the covered (u, v) plane with baseline in a .png file for visualization (8) 
                         Convert the observational plane to grayscale, re-arrange pixels, and save the baseline in .npy format

## Create Input data for GAN

1. image_noise.py: It defines the to add salt and paper noise to the ellipsoid image so that it can create noisy II data for the input to GAN. It also defines the function to add two 
                   images. Here, one image is a stellar source (fast-rotator) and the other is an II signal using this source and generated baselines.
2. generate_pspec.py: It calculates the power spectrum (the absolute value of the Fourier transform) of each noised image using salt and paper noise def in "image_noise.py". It also calculates the visibility 
                      for a  given baseline (power spectrum X baseline). Later, it normalizes the signal and adds both ground truth (ellipsoid image) and signal (visibility) side by side, which works as an input 
                      image for GAN. At the end, it saves the images in the train, test, and validation folders.
              
## Train the model with GAN for the given signal

1. main_def.py: This script prepares the set of definitions used for the GAN structure while training.
2. main_def_test.py: This script tests the definition used for the GAN structure in "main_def.py" while training.                           
3. gan_def, py: This script describes the GAN network as a generator and a Discriminator. Using these networks, GAN can train the model.
4. gan_def_test.py: This script test the "gan_def.py" and model is trained here.

# Enjoy!!!
