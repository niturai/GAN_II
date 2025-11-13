import matplotlib.pyplot as plt
from keras import backend as K
from main_def import main_fun 
from gan_def import gan_fun
import tensorflow as tf
import numpy as np
import datetime


fn = main_fun()
mfn = gan_fun()

fn.para(img_size=128)
mfn.gan_para(LAMBDA=100, OUTPUT_CHANNELS = 1, filtersize = 5, beta = 0.005, learning_rate = 2e-4, disc_train_iterations = 1)
BATCH_SIZE = 1                                                                                        #  The batch_size parameter specifies the number of samples that will be used in each iteration.
BUFFER_SIZE = 1400

# the spliting of image (sky and observational) 
sample_input, sample_real = fn.load('train/ellipse1.jpg')                                          # load the image to be trained in float32 tensor

# Converts a Keras model to dot format and save to a file.
generator = mfn.Generator()
tf.keras.utils.plot_model(generator, to_file='testing_image/generator.png', show_shapes=True, dpi=64) # structure of Generator according to the neuron

gen_output = generator(sample_input[tf.newaxis, ...], training=True)                                  # test the Generator to generate image using input signal (visibility) 
plt.imshow(gen_output[0, ...])
plt.savefig('testing_image/gen.png')
plt.show()

discriminator = mfn.Discriminator()
tf.keras.utils.plot_model(discriminator, to_file='testing_image/discriminator.png', show_shapes=True, dpi=64) # structure of Discriminator according to the neuron

disc_out = discriminator([sample_input[tf.newaxis, :], gen_output], training=False)                   # test the discrimator to predict the image (result) based on signal and generated imgae as input.
plt.imshow(disc_out[0, ..., -1], vmin=-10, vmax=10, cmap='RdBu_r')
plt.colorbar()
plt.savefig('testing_image/disc.png')
plt.show()

# Plots four images based on the model: Input, Ground truth, Prediction, and (optionally) difference in the predicted image and ground truth (if show_diff = True). 
example_input = sample_input[tf.newaxis, :]
example_target = sample_real[tf.newaxis, :]
mfn.generate_images(generator, example_input, example_target)

# check the helper function with train and test dataset
train_dataset = tf.data.Dataset.list_files(str('train/*.jpg'))                              # To create a dataset of all files matching a pattern
train_dataset = train_dataset.map(fn.load_image_train, num_parallel_calls=tf.data.AUTOTUNE) # apply load_image_train on each train_dataset
train_dataset = train_dataset.shuffle(BUFFER_SIZE)                                          # suffle the dataset
train_dataset = train_dataset.batch(BATCH_SIZE)                                             # for processing large amounts of data in a repeatable manner

try:
    test_dataset = tf.data.Dataset.list_files(str('test/*.jpg'))                            # to test the model
except tf.errors.InvalidArgumentError:
    test_dataset = tf.data.Dataset.list_files(str('val/*.jpg'))                             # to test the validity
test_dataset = test_dataset.map(fn.load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

mfn.fit(generator, discriminator, train_dataset, test_dataset, steps=100000)

counter = 0
for inp, tar in test_dataset.take(100):
    mfn.generate_images(generator, inp, tar, show_diff = True, sampling = True, save_image = True, counter = counter)
    counter += 1
    
