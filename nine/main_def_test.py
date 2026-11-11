import matplotlib.pyplot as plt
from main_def import main_fun
import tensorflow as tf
from scipy import fft
import numpy as np
import cv2

fn = main_fun()
fn.para(img_size=128)

# check the spliting of image (sky and observational) is working
# Load and split the concatenated JPEG
sample_input, sample_real = fn.load('../result/nine/val/ellipse3.jpg')

print("Before normalization:")
print("Input shape:", sample_input.shape)
print("Target shape:", sample_real.shape)

tf.print("Input min:", tf.reduce_min(sample_input))
tf.print("Input max:", tf.reduce_max(sample_input))
tf.print("Target min:", tf.reduce_min(sample_real))
tf.print("Target max:", tf.reduce_max(sample_real))

# Normalize explicitly
sample_input_norm, sample_real_norm = fn.normalize(sample_input, sample_real)

print("\nAfter normalization:")
tf.print("Input min:", tf.reduce_min(sample_input_norm))
tf.print("Input max:", tf.reduce_max(sample_input_norm))
tf.print("Target min:", tf.reduce_min(sample_real_norm))
tf.print("Target max:", tf.reduce_max(sample_real_norm))

# Display normalized images
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(tf.squeeze(sample_input_norm).numpy(), cmap="gray", vmin=-1, vmax=1)
plt.title("Normalized input")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(tf.squeeze(sample_real_norm).numpy(), cmap="gray", vmin=-1, vmax=1)
plt.title("Normalized target")
plt.colorbar()
plt.tight_layout()
plt.savefig('../result/nine/ft/normalize.jpg')
#plt.show()
plt.close()

print(np.shape(sample_input))
pic = 0
for i in [sample_input, sample_real]:
    plt.imshow(i)
    plt.savefig('../result/nine/testing_image/sep' + str(pic) + '.png')
    pic += 1
    #plt.show()
    plt.close()

# Check if random_jitter is working
obs_jit, sky_jit = fn.random_jitter(sample_input, sample_real)
for i in [obs_jit, sky_jit]:
    plt.imshow(i)
    plt.savefig('../result/nine/testing_image/jit' + str(pic) + '.png')
    pic += 1
    #plt.show()
    plt.close()
    
plt.figure(figsize=(6, 6))
for i in range(4):                                                               # flip the images 4 times
    jit_obs, jit_sky = fn.random_jitter(sample_input, sample_real)
    plt.subplot(2, 2, i+1)
    plt.imshow(jit_sky)
    plt.axis('off')
    
plt.savefig('../result/nine/testing_image/jit_sky' + str(pic) + '.png')
#plt.show()
plt.close()

# Display the sky image with salt and pepper noise
spimage = fn.Saltandpepper(sample_real, 0.005, display_img=True)
plt.imshow(spimage)
plt.savefig('../result/nine/testing_image/salt_pepper.png')
#plt.show()
plt.close()

# Downsampling
down_model = fn.downsample(3, 4)
down_result = down_model(tf.expand_dims(tf.cast(sample_input, float), 0))
print('down sampling result of an input image, observed signal', down_result.shape)

# Upsampling
up_model = fn.upsample(3, 4, apply_dropout=True)
up_result = up_model(down_result)
print('up sampling result of down sampled image', up_result.shape)


