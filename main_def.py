from keras import backend as K
import tensorflow as tf
from scipy import fft
import numpy as np

class main_fun():
      """
      The functions to load data, resize, crop and normalization of data.
      """
      def para(self, img_size):
          self.img_size = img_size
      
      def load(self, image_file):
          """
          Loads the concatanated image files and separates them. 

          Input:
          ------
          image-file : the file name to separate

          Output:
          ------
          Observed signla on observational plane (u, v) and Sky image (ground truth)
          """
          
          image = tf.io.read_file(image_file)                                     # read the jpeg file as a tensor of dtype "string"
          image = tf.io.decode_jpeg(image)                                        # convert or decode the jpeg to unit8 tensor
          
          # Split the combined image (object in sky and on observational plane) into two different image
          w = tf.shape(image)[1]                                                  # as the image has been added horizontally [1], [0] if vertically
          w = w // 2                                                              # equal division of width of original image
          real_image = image[:, :w, :]                                            # first half (w) is sky image
          input_image = image[:, w:, :]                                           # second half (w) is observed image (power spectrum)
          
          
          # convert both input image from unit8 to float32 tensor using tf.cast
          real_image = tf.cast(real_image, tf.float32)
          input_image = tf.cast(input_image, tf.float32)
          
          return input_image, real_image
                    
      def resize(self, input_image, real_image, height, width):
          """
          Resize the both signal and object image to change the shape of an array without 
          changing its data, using Nearest-neighbor interpolation method.
          
          Input:
          
          sky_image : the sky image
          obs_image : the observed signal
          height : Height of the image
          width : Width of the image
          
          Output:
          
          Resized image according to height and width.
          """
          
          input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
          real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
          
          return input_image, real_image
                    
      def random_crop(self, input_image, real_image):
          """
          crop the both input image according to image size.
          
          Input:
          
          sky_image : the sky image
          obs_image : the observed image
          
          Output:
          
          Croped sky and observed image according to image size.
          """
          
          stacked_image = tf.stack([input_image, real_image], axis=0)                                    # Stacks the given tensor [sky_image, obs_image] into 1 higher rank tensor
          cropped_image = tf.image.random_crop(stacked_image, size=[2, self.img_size, self.img_size, 1])    # Randomly crops a tensor to a given image size into 2 image [sky_image, obs_image].
          
          return cropped_image[0], cropped_image[1]
                    
      def normalize(self, input_image, real_image):
          """
          Normalizing the images to [-1, 1]. This is done to ensure that the input data has a mean of 0 and a standard deviation of 1,
          which can help improve the performance of machine learning models.
          
          Input:
          ------
          sky_image : the sky image
          obs_image : the observed image
          
          Output:
          ------
          Normalize the image according to image size.
          """
          
          input_image = (input_image / (self.img_size - 0.5)) - 1                                       # maximum element number in sky_image is 255 and minimum is 0
          real_image = (real_image / (self.img_size - 0.5)) - 1                                         # maximum element number in obs_image is 255 and minimum is 0
         
          return input_image, real_image
          
      @tf.function()   
      def random_jitter(self, input_image, real_image):
          """
          Jittering the images.
          Input:
          
          sky_image : the sky image
          obs_image : the observed image
          
          Output:
          
          Jittered images according to image size.
          """
          input_image, real_image = self.resize(input_image, real_image, int(self.img_size + self.img_size*0.0625), 
                                             int(self.img_size + self.img_size*0.0625))                   # Resizing by 6.25%, so the shape of image is (136, 136)
          input_image, real_image = self.random_crop(input_image, real_image)                             # Random cropping back and image size is (128, 128) again
          
          if tf.random.uniform(()) > 0.5:                                                                 # the half set of images as signal is squared visibility and phase remains in visibility
             input_image = tf.image.flip_left_right(input_image)                                          # flip images from left to right
             real_image = tf.image.flip_left_right(real_image)
          
          return input_image, real_image
          
      def load_image_train(self, image_file):
          """
          Loads the concatanated image files and separates, 
          then jitter (flipping half images as phase reconstruction exist in visibility) 
          and normalize them. 

          Input:
          ------
          image-file : the file name to separate

          Output:
          ------
          Sky image and observed image (ground truth) after jittering and normalization.
          It used for training the model.
          """
          
          input_image, real_image = self.load(image_file)
          input_image, real_image = self.random_jitter(input_image, real_image)
          input_image, real_image = self.normalize(input_image, real_image)
          
          return input_image, real_image
          
      def load_image_test(self, image_file):
          """
          Loads the concatanated image files and separates, 
          then resize and normalize them (for testing model, flipping is not done as the testing will be done). 

          Input:
          ------
          image-file : the file name to separate

          Output:
          ------
          Sky image and observed image (ground truth) after resize and normalization.
          It used for testing the model.
          """
          
          input_image, real_image = self.load(image_file)
          input_image, real_image = self.resize(input_image, real_image, self.img_size, self.img_size)
          input_image, real_image = self.normalize(input_image, real_image)
          
          return input_image, real_image
          
      def Saltandpepper(self, image, prob, display_img=False):
          """
          Add salt and pepper noise to image using tf.where().
          An extra values (from 0 to 1) are added in elements where conditions apply in tf.where() 
          
          Input :
          image : the input image
          prob : probability of the noise
          
          Return : an image with salt and pepper noise
          """
          
          if display_img:
             random_values = tf.random.uniform(shape=image.shape)                                           # Outputs random values (0, 1) from a uniform distribution in shape (128, 128, 1).
          else:
             random_values = tf.random.uniform(shape=image[0,:,:,:].shape)                                  # Outputs random values (0, 1) from a uniform distribution in shape (128, 1).
          
          image = tf.where(random_values < (prob/2), self.img_size - 1., image)                             # tf.where(condition, a, b), a has the same dimension as condition has.
          image = tf.where(1 - random_values < (prob/2), 0., image)
          return image
          
      def ff2d_diff(self, image1, image2, sampling=False):
          """
          calculate the difference between two images in the Fourier Plane (observed signal)
          
          Input :
          image1 : the first image
          image2 : the second image
          
          Output :
          absolute difference between Fourier-shifted 2D FFT 
          """   
          
          image1_fft = np.abs(fft.fftshift(fft.fft2(fft.fftshift(image1))))                                 # visibility (II signal)
          image2_fft = np.abs(fft.fftshift(fft.fft2(fft.fftshift(image2))))
         
          output = (image1_fft - image2_fft)
          
          if sampling:
             sampling_mask = np.load("base_npy/base.npy")
             output = np.multiply(output, sampling_mask)                                                    # visibility (II signal) for the given baselines
             
          return output
      
      def downsample(self, filters, size, apply_batchnorm=True):
          """
          To scale down the network
          """
          
          initializer = tf.random_normal_initializer(0.0, 0.02)                        # Initializer that generates tensors with a normal distribution (mean = 0.0, stddev = 0.02)
          result = tf.keras.Sequential()                                               # helps to form a cluster of a layer that is linearly stacked into tf.keras.Model
          result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                     kernel_initializer=initializer, data_format="channels_last",
                     use_bias=False))                                                  # spatial convolution over images Used to scale down the input shape

          if apply_batchnorm:
             result.add(tf.keras.layers.BatchNormalization())                          # Layer that normalizes its inputs.
    
          result.add(tf.keras.layers.LeakyReLU())                                      # Leaky version of a Rectified Linear Unit to activate neurons

          return result
            
      def upsample(self, filters, size, apply_dropout=False):
          """
          To scale up the network
          """
          
          initializer = tf.random_normal_initializer(0., 0.02)                        # Initializer that generates tensors with a normal distribution, (mean = 0.0, stddev = 0.02 here)
          result = tf.keras.Sequential()                                              # Sequential provides training and inference features on this model.
          result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                     padding='same', kernel_initializer=initializer,
                     data_format="channels_last", use_bias=False))                    # spatial deconvolution over images Used to scale up the input shape
          result.add(tf.keras.layers.BatchNormalization())                            # normalize the mini batches

          if apply_dropout:
             result.add(tf.keras.layers.Dropout(0.5))                                 # randomly selected neurons are ignored during training

          result.add(tf.keras.layers.ReLU())                                          # Rectified Linear Unit activation function to activate neurons
 
          return result 

