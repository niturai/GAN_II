import numpy as np


class ImageNoise():
   """
   Add salt and pepper noise to image
   """
   def sap_noise(self, image, prob):
         """
         Add salt and pepper noise to image
         image : the uploaded image
         prob : Salt and Pepper Noise probability
         return : an output with grayscale according to prob
         """
         output = image.copy()                               # image shape is (N, N) = (512, 512)
         if len(image.shape) == 2:
            black = 0
            white = 1
         else:
            raise ValueError('This image has multiple channels, which is not supported.')
         probs = np.random.random(output.shape[:2])          # calculate the random prob (0, 1) for each pixel (N, N)
         output[probs < (prob / 2)] = black                  # pixel is black if 
         output[probs > 1-(prob / 2)] = white                # pixel is white if
         return output
