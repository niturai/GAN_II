import matplotlib.pyplot as plt
from keras import backend as K
from main_def import main_fun
from gan_def import gan_fun
import tensorflow as tf
import numpy as np
import datetime
import pickle
import os

print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))

# GENERAL PARAMETERS
fn = main_fun()
fn.para(img_size=128)

BATCH_SIZE = 1
BUFFER_SIZE = 1400
LAMBDA = 100
OUTPUT_CHANNELS = 1
FILTERSIZE = 5
BETA = 0.005
LEARNING_RATE = 2e-4
STEPS = 100000

DISCRIMINATOR_REPETITIONS = [1, 2, 3] # DISCRIMINATOR REPETITION VALUES
sample_input, sample_real = fn.load('../result/six/val/ellipse3.jpg')# LOAD EXAMPLE IMAGE

# TRAINING DATASET
train_dataset = (
    tf.data.Dataset.list_files(
        '../result/six/train/*.jpg',
        shuffle=True
    )
    .map(
        fn.load_image_train,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# VALIDATION DATASET
validation_dataset = (
    tf.data.Dataset.list_files(
        '../result/six/val/*.jpg',
        shuffle=False
    )
    .map(
        fn.load_image_test,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# TEST DATASET
test_dataset = (
    tf.data.Dataset.list_files(
        '../result/six/test/*.jpg',
        shuffle=False
    )
    .map(
        fn.load_image_test,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# STORE LOSSES FROM ALL THREE EXPERIMENTS
all_logs = {}

# RUN THREE EXPERIMENTS
for disc_rep in DISCRIMINATOR_REPETITIONS:
    print("\n", flush=True)
    print("=" * 80, flush=True)
    print(f"STARTING EXPERIMENT: Discriminator Repetition = {disc_rep}", flush=True)
    print("=" * 80, flush=True)

    mfn = gan_fun()                      # Create a NEW gan_fun object
 
    # Set GAN parameters
    mfn.gan_para(LAMBDA=LAMBDA, OUTPUT_CHANNELS=OUTPUT_CHANNELS, 
       filtersize=FILTERSIZE, beta=BETA, learning_rate=LEARNING_RATE, 
       disc_train_iterations=disc_rep)

    generator = mfn.Generator()          # Create FRESH Generator    
    discriminator = mfn.Discriminator()  # Create FRESH Discriminator
    
    os.makedirs('../result/six/testing_image', exist_ok=True)

    # Optional: save model diagrams only once
    if disc_rep == DISCRIMINATOR_REPETITIONS[0]:
        tf.keras.utils.plot_model(generator, to_file='../result/six/testing_image/generator.png',
                show_shapes=True, dpi=64)

        tf.keras.utils.plot_model(discriminator, to_file='../result/six/testing_image/discriminator.png',
                show_shapes=True, dpi=64)

    # Test Generator before training
    gen_output = generator(sample_input[tf.newaxis, ...], training=False)

    if disc_rep == DISCRIMINATOR_REPETITIONS[0]:
        plt.figure()
        plt.imshow(gen_output[0, ...])
        plt.savefig('../result/six/testing_image/gen.png',
            dpi=300, bbox_inches='tight')
        plt.close()

    # Test Discriminator before training
    disc_out = discriminator([sample_input[tf.newaxis, :], gen_output], training=False)
    if disc_rep == DISCRIMINATOR_REPETITIONS[0]:
        plt.figure()
        plt.imshow(disc_out[0, ..., -1], vmin=-10, vmax=10, cmap='RdBu_r')
        plt.colorbar()
        plt.savefig('../result/six/testing_image/disc.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Test generate_images()
    example_input = sample_input[tf.newaxis, :]
    example_target = sample_real[tf.newaxis, :]

    if disc_rep == DISCRIMINATOR_REPETITIONS[0]:
        mfn.generate_images(generator, example_input, example_target)

    # TRAIN
    output_dir = (f"../result/six/testing_image/disc_rep_{disc_rep}")
    os.makedirs(output_dir, exist_ok=True)
    history_df = mfn.fit(generator, discriminator, train_dataset, validation_dataset, steps=STEPS, output_dir=output_dir)

    # STORE THIS EXPERIMENT
    run_name = f"disc_rep_{disc_rep}"
    all_logs[run_name] = history_df

    print("\n", flush=True)
    print(f"Finished discriminator repetition = {disc_rep}", flush=True)

    # SAVE TRAINED MODELS
    print("Training completed. Starting model saving...", flush=True)
    generator.save(f"../result/six/testing_image/generator_disc_rep_{disc_rep}.keras")
    print("Generator saved.", flush=True)
    discriminator.save(f"../result/six/testing_image/discriminator_disc_rep_{disc_rep}.keras")
    print("Discriminator saved.", flush=True)
    print("Saving training history...", flush=True)

    # SAVE TRAINING LOGS
    with open("logs.pkl", "wb") as f:
        pickle.dump(all_logs, f)
    
    print("Training history saved.", flush=True)   
    test_output_dir = os.path.join(output_dir, "test")
    os.makedirs(test_output_dir, exist_ok=True)
    print("Starting test-image generation...", flush=True)
    counter = 0
    for inp, tar in test_dataset.take(100):
        mfn.generate_images(generator, inp, tar, show_diff=True, sampling=True, save_image=True, counter=counter, output_dir=test_output_dir)
        counter += 1
        
        if counter % 10 == 0:
           print(f"Generated {counter} test images.", flush=True)

    print("Test-image generation completed.", flush=True)
        
