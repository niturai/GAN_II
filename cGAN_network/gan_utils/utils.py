import os
import tensorflow as tf
from keras import layers
import numpy as np


class GanUtils:
    """
    Utility functions for loading, preprocessing, and augmenting GAN input data.

    Notes
    -----
    - All image preprocessing is Keras-graph safe (uses tf.signal or keras.layers).
    - Normalization: images are scaled to [-1, 1].
    - Compatible with both tf.data pipelines and Keras Functional API.
    """

    def __init__(self, img_size: int = 128):
        self.img_size = img_size
        self._resize_layer = layers.Resizing(img_size, img_size, interpolation="nearest")
        self._flip_layer = layers.RandomFlip("horizontal")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load(self, image_file: str):
        """
        Load concatenated image file and split into (observed, real).
        Input format: [real_image | input_image] concatenated along width.
        """
        image = tf.io.read_file(image_file)
        image = tf.io.decode_jpeg(image, channels=1)

        w = tf.shape(image)[1] // 2
        real_image = image[:, :w, :]
        input_image = image[:, w:, :]

        real_image = tf.cast(real_image, tf.float32)
        input_image = tf.cast(input_image, tf.float32)
        return input_image, real_image

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def resize(self, input_image, real_image, height, width):
        """Resize input and real images to (height, width)."""
        resize_layer = layers.Resizing(height, width, interpolation="nearest")
        return resize_layer(input_image), resize_layer(real_image)

    def random_crop(self, input_image, real_image):
        """Randomly crop to (img_size, img_size)."""
        stacked = tf.stack([input_image, real_image], axis=0)
        cropped = tf.image.random_crop(
            stacked, size=[2, self.img_size, self.img_size, tf.shape(input_image)[-1]]
        )
        return cropped[0], cropped[1]

    def normalize(self, input_image, real_image):
        """Normalize pixel values to [-1, 1]."""
        input_image = (input_image / 127.5) - 1.0
        real_image = (real_image / 127.5) - 1.0
        return input_image, real_image

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------
    @tf.function
    def random_jitter(self, input_image, real_image):
        """
        Apply resize (slight upscale), random crop, and random flip.
        """
        upscale = int(self.img_size * 1.0625)  # +6.25%
        input_image, real_image = self.resize(input_image, real_image, upscale, upscale)
        input_image, real_image = self.random_crop(input_image, real_image)

        if tf.random.uniform(()) > 0.5:
            input_image = self._flip_layer(input_image)
            real_image = self._flip_layer(real_image)

        return input_image, real_image

    # def load_image_train(self, image_file):
    #     """Load and preprocess training image pair."""
    #     input_image, real_image = self.load(image_file)
    #     input_image, real_image = self.random_jitter(input_image, real_image)
    #     input_image, real_image = self.normalize(input_image, real_image)
    #     return input_image, real_image
    def load_image_train(self, image_file):
        input_image, real_image = self.load(image_file)  # decode_jpeg(channels=1)
        input_image, real_image = self.random_jitter(input_image, real_image)
        input_image, real_image = self.normalize(input_image, real_image)
        # Set static shapes to avoid shape surprises later
        input_image = tf.ensure_shape(input_image, [self.img_size, self.img_size, 1])
        real_image  = tf.ensure_shape(real_image,  [self.img_size, self.img_size, 1])
        return input_image, real_image


    def load_image_test(self, image_file):
        """Load and preprocess test image pair (no jitter)."""
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.resize(input_image, real_image, self.img_size, self.img_size)
        input_image, real_image = self.normalize(input_image, real_image)
        return input_image, real_image

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------
    # def salt_and_pepper(self, image, prob=0.005):
    #     """
    #     Add salt & pepper noise.

    #     Parameters
    #     ----------
    #     image : tf.Tensor
    #         Input image tensor (H, W, C).
    #     prob : float
    #         Probability of flipping a pixel to black or white.
    #     """
    #     random_values = tf.random.uniform(shape=tf.shape(image), dtype=tf.float32)
    #     noisy = tf.where(random_values < (prob / 2), 0.0, image)              # pepper
    #     noisy = tf.where(random_values > 1 - (prob / 2), 255.0, noisy)        # salt
    #     return noisy

    # ------------------------------------------------------------------
    # Fourier utilities
    # ------------------------------------------------------------------
    def ff2d_diff(self, image1, image2, sampling=False, base_path=''):
        """
        Compute difference in Fourier domain between two images.
        Uses tf.signal (graph-safe).

        Parameters
        ----------
        image1, image2 : tf.Tensor
            Input images (H, W).
        sampling : bool
            If True, multiply difference by baseline mask.

        Returns
        -------
        tf.Tensor
            Difference in Fourier domain (H, W).
        """
        image1_fft = tf.signal.fft2d(tf.cast(image1, tf.complex64))
        image1_fft = tf.signal.fftshift(image1_fft)
        image1_fft = tf.abs(image1_fft)

        image2_fft = tf.signal.fft2d(tf.cast(image2, tf.complex64))
        image2_fft = tf.signal.fftshift(image2_fft)
        image2_fft = tf.abs(image2_fft)

        diff = image1_fft - image2_fft

        if sampling:
            sampling_mask = np.load(os.path.join(base_path, "base.npy"))
            diff = diff * tf.convert_to_tensor(sampling_mask, dtype=tf.float32)

        return diff

    # ------------------------------------------------------------------
    # Network blocks
    # ------------------------------------------------------------------
    def downsample(self, filters, size, apply_batchnorm=True):
        """Downsampling block for U-Net / GAN."""
        initializer = tf.random_normal_initializer(0.0, 0.02)
        result = tf.keras.Sequential()
        result.add(
            layers.Conv2D(
                filters,
                size,
                strides=2,
                padding="same",
                kernel_initializer=initializer,
                use_bias=False,
            )
        )
        if apply_batchnorm:
            result.add(layers.BatchNormalization())
        result.add(layers.LeakyReLU())
        return result

    def upsample(self, filters, size, apply_dropout=False):
        """Upsampling block for U-Net / GAN."""
        initializer = tf.random_normal_initializer(0.0, 0.02)
        result = tf.keras.Sequential()
        result.add(
            layers.Conv2DTranspose(
                filters,
                size,
                strides=2,
                padding="same",
                kernel_initializer=initializer,
                use_bias=False,
            )
        )
        result.add(layers.BatchNormalization())
        if apply_dropout:
            result.add(layers.Dropout(0.5))
        result.add(layers.ReLU())
        return result

class SaltAndPepper(tf.keras.layers.Layer):
    def __init__(self, prob=0.005, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob

    def call(self, inputs, training=None):
        if not training:   # Only apply noise during training
            return inputs

        random_values = tf.random.uniform(shape=tf.shape(inputs), dtype=tf.float32)
        noisy = tf.where(random_values < (self.prob / 2), 0.0, inputs)          # pepper
        noisy = tf.where(random_values > 1 - (self.prob / 2), 255.0, noisy)     # salt
        return noisy