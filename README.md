## GAN_II
The image reconstruction on optical wavelengths with high precision has always been challenging for astronomers. Here, we present the application of a  Generative Adversarial Network (GAN) network for Intensity Interferometry and do the image reconstruction for a fast-rotator.

The Python files follow the information of model of star and baseline configuration as follows:

# To create N number of Images....

1. ellipse.py: Define a function to create N number of 2 dimensional (x, y) grids to draw contour map of ellipse with different parameters (radius, inclination, position angle, axis ratio)
2. generate_image.py: The parameters of ellipse is defined here and save them in .jpg format

# To simulate the Telescope and Baselines Configuration.......

1. aperture.py: It define the shape of the aperture for each telescopes and visualize it on observational plane. It also define the a function to create the array of baseline's name and length on  
                observational plane.
2. obstime.py: It calculate the number of steps at each observational night due to observational intervals. It also calculate the start and end time time after each observational interval. There is 
               another definitions to calculate an array of all Julian days for all given observational interval.
3. obspoint.py: Since everything is calculated in radian. There are definitions to convert the degree and hour to radian. There is also another function which calculate the change in baseline due to 
                earth rotation
4. generate_baseline.py: It generate the variational baseline with time, which is used to simulate the II signal for any source and telescope's array. However, we need to define some parameter here, 
                         exa: (1) Array of starting times of the observational day. (2) Array of ending times for the observational day (3) calculate the number of steps for given observational interval 
                         (average time) so the array of Julian time (4) Define the telescope's position and name. Then get the baseline's name and length. (5) Define the position of of source in the sky 
                         coordinate. (6) Calculate the variational baseline according to the earth's rotation. (7) Plot the covered (u, v) plane with baseline in a .png file for visualization (8) 
                         Convert the observational plane to grayscale, re-arrange pixels, and save the baseline in .npy format

# Create Input data for GAN

1. image_noise.py: It define the to add salt and paper noise to the ellipsoid image so that it can create noisy II data for the input to GAN. It also define the function to add two 
                   image. Here, one image is stellar source (fast-rotator) and other is II signal using this source and generated baselines.
2. generate_pspec.py: It calculate the power spectrum (the absolute value of Fourier transform) of each noised image using salt and paper noise def in "image_noise.py". It also calculate the visibility 
                      for given baseline (power spectrum X baseline). Later it normalise the signal and add both ground truth (ellipsoid image) and signal (visibility) side by side, which works as input 
                      image for GAN. At the end it saves the images in the train, test and validation folders.
              
# Train the model with GAN for the given signal

1. main_def.py: This script prepares the set of definitions used for the GAN structure while training.
2. main_def_test.py: This script tests the definition used for the GAN structure in "main_def.py" while training.                           
3. gan_def,py: This script describes the GAN network as a generator and Discriminator. Using these network GAN can trained the model.
4. gan_def_test.py: This script test the "gan_def.py" and model is trained here.

## Enjoy!!!
