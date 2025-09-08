# Data Preparation

This module contains the pipeline for simulating sparsely sampled Intensity 
Interferometry signal.

The goal is to produce ML-ready datasets that can be used by the cGAN network. 
---



## Workflow

1. **Generate ellipsoids (representing images of fast rotating stars)**

   ```bash
   python generate_ellipsoids.py --path /path/to/output_dir/
   ```
   

2. **Generate telescope and baseline configuration**

   ```bash
   python generate_baseline.py --path /path/to/output_dir/ --ntele 4
   ```

3. **Combine (1) and (2) to generate input data for cGAN**

   ```bash
   python generate_powerspec.py --path /path/to/output_dir/
   ```
---

## Utils

* ```aperture.py```: Aperture class defines the shape of the aperture for each telescope and visualizes it on the observational plane. It also defines a function to create an array of baseline' names and length on  the observational plane.

* ```ellipsoid.py```: Ellipsoid class defines a function to create N number of 2-dimensional (x, y) grids to draw a contour map of an ellipse with different parameters (radius, inclination, position angle, axis ratio)

* ```image_merge.py```: ImageMerge class defines how the {input, output} data points are concatenated horizontally for the cGAN input. 

* ```image_noise.py```: ImageNoise class defines how sald and pepper noise is added. 

* ```obspoint.py```: ObsPoint class defines the the conversion from degrees and hours to radians. Here we also calculate the change in baseline due to earths rotation. 

* ```obstime.py```: ObsTime class calculates the number of steps at each observational night due to observational intervals. It also calculates the start and end time after each observational interval. There is another definition to calculate an array of all Julian days for all given observational intervals.


## Notes

* Make sure to use the same out path/to/output_dir for the best compatability