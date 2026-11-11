import numpy as np


class Pimage():
      """
      Add the ellipsoid and its power spectrum also add some noise
      """
      
      def add_image(self, img_a, img_b):
          """
          combine two color image in form of nd_array side by side
          """
          ha, wa = img_a.shape[:2]                            # hight and width of img_a
          hb, wb = img_b.shape[:2]                            # hight and width of img_b
          max_height = np.max([ha, hb])                       # maximum height
          total_width = wa+wb                                 # total width
          new_img = np.zeros(shape=(max_height, total_width)) # new image of shape, maximum height and total width
          new_img[:ha, :wa] = img_a                           # insert first image
          new_img[:hb, wa:wa+wb] = img_b                      # insert second image
          return new_img
          
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
