"""
GAN Training and Evaluation Script
----------------------------------

This script trains a GAN model using TensorFlow/Keras on astronomical images.
It supports GPU execution if available, and generates visualization outputs 
for generator and discriminator models, as well as sample predictions.
"""

import os
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] Using {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"[WARNING] GPU setup failed: {e}")
else:
    print("[INFO] Running on CPU.")

from gan_utils.utils import GanUtils
from model.gan import CGAN


# -------------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------------
def main(base_path: str):
    """
    Main training pipeline.

    Args:
        base_path (str): Base path containing `train/`, `test/` or `val/` directories.
    """
    # Verify GPUs are used
    print("Is GPU available?", tf.config.list_physical_devices("GPU"))
    print("TensorFlow device:", tf.test.gpu_device_name())

    # ---------------------------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------------------------
    fn = GanUtils(img_size=128)
    mfn = CGAN()

    BATCH_SIZE = 1      # Number of samples per training iteration
    BUFFER_SIZE = 140  # Shuffle buffer size for dataset

    mfn.gan_para(
        LAMBDA=100,
        OUTPUT_CHANNELS=1,
        filtersize=5,
        beta=0.005,
        learning_rate=2e-4,
        disc_train_iterations=1,
        base_path=base_path, 
        batch_size=BATCH_SIZE
    )

    # ---------------------------------------------------------------------
    # DATA LOADING AND VISUALIZATION
    # ---------------------------------------------------------------------
    # Load one sample image for sanity check
    sample_input, sample_real = fn.load(os.path.join(base_path, "output/train/ellipse_001802.jpg"))
    os.makedirs("testing_image", exist_ok=True)

    # ---------------------------------------------------------------------
    # GENERATOR MODEL
    # ---------------------------------------------------------------------
    generator = mfn.Generator()
    tf.keras.utils.plot_model(
        generator, to_file=os.path.join(base_path, "testing_image/generator.png"), show_shapes=True, dpi=64
    )

    gen_output = generator(sample_input[tf.newaxis, ...], training=True)
    plt.imshow(gen_output[0, ...])
    plt.title("Generator Output")
    plt.savefig(os.path.join(base_path, "testing_image/gen.png"))
    plt.show()

    # ---------------------------------------------------------------------
    # DISCRIMINATOR MODEL
    # ---------------------------------------------------------------------
    discriminator = mfn.Discriminator()
    tf.keras.utils.plot_model(
        discriminator, to_file=os.path.join(base_path, "testing_image/discriminator.png"), show_shapes=True, dpi=64
    )

    disc_out = discriminator([sample_input[tf.newaxis, :], gen_output], training=False)
    plt.imshow(disc_out[0, ..., -1], vmin=-10, vmax=10, cmap="RdBu_r")
    plt.colorbar()
    plt.title("Discriminator Output")
    plt.savefig(os.path.join(base_path, "testing_image/disc.png"))
    plt.show()

    # ---------------------------------------------------------------------
    # EXAMPLE VISUALIZATION
    # ---------------------------------------------------------------------
    example_input = sample_input[tf.newaxis, :]
    example_target = sample_real[tf.newaxis, :]
    mfn.generate_images(generator, example_input, example_target)

    # ---------------------------------------------------------------------
    # DATASET PREPARATION
    # ---------------------------------------------------------------------
    train_dataset = tf.data.Dataset.list_files(os.path.join(base_path, "output/train/*.jpg"))
    train_dataset = train_dataset.map(fn.load_image_train)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    # Use "test/" if exists, else fall back to "val/"
    try:
        test_dataset = tf.data.Dataset.list_files(os.path.join(base_path, "output/test/*.jpg"))
    except tf.errors.InvalidArgumentError:
        test_dataset = tf.data.Dataset.list_files(os.path.join(base_path, "output/val/*.jpg"))

    test_dataset = test_dataset.map(fn.load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)


    # ---------------------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------------------
    mfn.fit(generator, discriminator, train_dataset, test_dataset, steps=100000)


    # Generate predictions for a subset of test images
    counter = 0
    for inp, tar in test_dataset.take(20):
        mfn.generate_images(
            generator,
            inp,
            tar,
            show_diff=True,
            sampling=True,
            save_image=True,
            counter=counter,
        )
        counter += 1


# -------------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a GAN model on astronomical images."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Base path containing base.npy and output/train/ datasets.",
    )
    args = parser.parse_args()

    main(args.path)
