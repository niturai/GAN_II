import matplotlib.pyplot as plt
from main_def import main_fun
import tensorflow as tf
from scipy import fft
import numpy as np
import cv2

fn = main_fun()
fn.para(img_size=128)

# check the signal on observational plane and captured one using baseline of a stellar object
base = np.load("base_npy/base.npy") 
train_image = np.load("ellip_npy/ellipse1612.npy")
train_image = cv2.resize(train_image, dsize=(128, 128), interpolation=cv2.INTER_AREA)
print('base shape', np.shape(base))
print('image shape', np.shape(train_image))
fft = np.abs(fft.fftshift(fft.fft2(fft.fftshift(train_image))))
# the signal on observational plane
fft1 = fft/fft.max()
img = plt.imshow(fft1)
plt.colorbar(img)
plt.gca().set_aspect('equal')
plt.axis('off')
plt.savefig('ft/ft.jpg')
plt.show()

plt.close()
img1 = plt.imshow(np.log(fft1))
plt.colorbar(img1)
plt.gca().set_aspect('equal')
plt.axis('off')
plt.savefig('ft/ft_log.jpg')
plt.show()

plt.close()
# the signal captured with baseline on observational plane
output = np.multiply(fft, base) 
output = output/output.max()
img2 = plt.imshow(output)
plt.colorbar(img2)
plt.gca().set_aspect('equal')
plt.axis('off')
plt.savefig('ft/ft_base.jpg')
plt.show()

plt.close()
# the signal captured with baseline on log observational plane
img3 = plt.imshow(np.log10(output+1e-7))
plt.colorbar(img3)
plt.gca().set_aspect('equal')
plt.axis('off')
plt.savefig('ft/ft_log_base.jpg')
plt.show()



plt.close()
# The full fourier transformation on ground of stellar object and the captured one with baselines
plt.subplot(1,2,1)
plt.imshow(fft)
plt.subplot(1,2,2)
plt.imshow(output)
#plt.show()


# check the spliting of image (sky and observational) is working
sample_input, sample_real = fn.load('train/ellipse1612.jpg')                         # load the image to be trained in float32 tensor
print(np.shape(sample_input))
pic = 0
for i in [sample_input, sample_real]:
    plt.imshow(i)
    plt.savefig('testing_image/sep' + str(pic) + '.png')
    pic += 1
    plt.show()

# Check if random_jitter is working
obs_jit, sky_jit = fn.random_jitter(sample_input, sample_real)
for i in [obs_jit, sky_jit]:
    plt.imshow(i)
    plt.savefig('testing_image/jit' + str(pic) + '.png')
    pic += 1
    plt.show()
    
plt.figure(figsize=(6, 6))
for i in range(4):                                                               # flip the images 4 times
    jit_obs, jit_sky = fn.random_jitter(sample_input, sample_real)
    plt.subplot(2, 2, i+1)
    plt.imshow(jit_sky)
    plt.savefig('testing_image/jit_sky' + str(pic) + '.png')
    plt.axis('off')
plt.show()

# Display the sky image with salt and pepper noise
spimage = fn.Saltandpepper(sample_real, 0.005, display_img=True)
plt.imshow(spimage)
plt.savefig('testing_image/salt_pepper.png')
plt.show()

# Downsampling
down_model = fn.downsample(3, 4)
down_result = down_model(tf.expand_dims(tf.cast(sample_input, float), 0))
print('down sampling result of an input image, observed signal', down_result.shape)

# Upsampling
up_model = fn.upsample(3, 4, apply_dropout=True)
up_result = up_model(down_result)
print('up sampling result of down sampled image', up_result.shape)

