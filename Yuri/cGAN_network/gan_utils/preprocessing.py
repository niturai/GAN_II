"""
Fourier Analysis and Preprocessing Script
-----------------------------------------

This script performs Fourier analysis on input images and tests utility
functions (splitting, jitter, noise, downsampling, upsampling) from main_fun.

Usage:
    python utils/fourier_analysis.py --base_path ./data
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from scipy import fft

from utils import GanUtils


def main(base_path: str):
    """Run Fourier analysis and preprocessing checks."""
    fn = GanUtils()
    fn.para(img_size=128)

    # ---------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------
    base = np.load(os.path.join(base_path, "base.npy"))
    train_image = np.load(os.path.join(base_path, "ellip_npz/ellipse_001352.npz"))
    train_image = cv2.resize(train_image, dsize=(128, 128), interpolation=cv2.INTER_AREA)

    print("Base shape:", np.shape(base))
    print("Image shape:", np.shape(train_image))

    # ---------------------------------------------------------------------
    # Fourier transform
    # ---------------------------------------------------------------------
    ft_image = np.abs(fft.fftshift(fft.fft2(fft.fftshift(train_image))))
    ft_norm = ft_image / ft_image.max()

    # Save linear scale FT
    plt.imshow(ft_norm)
    plt.colorbar()
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.savefig("ft/ft.jpg")
    plt.show()
    plt.close()

    # Save log scale FT
    plt.imshow(np.log(ft_norm + 1e-7))
    plt.colorbar()
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.savefig("ft/ft_log.jpg")
    plt.show()
    plt.close()

    # Captured signal with baseline
    captured = np.multiply(ft_image, base)
    captured = captured / captured.max()

    plt.imshow(captured)
    plt.colorbar()
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.savefig("ft/ft_base.jpg")
    plt.show()
    plt.close()

    # Log captured signal
    plt.imshow(np.log10(captured + 1e-7))
    plt.colorbar()
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.savefig("ft/ft_log_base.jpg")
    plt.show()
    plt.close()

    # Compare full FT and captured
    plt.subplot(1, 2, 1)
    plt.imshow(ft_image)
    plt.subplot(1, 2, 2)
    plt.imshow(captured)
    plt.savefig("ft/ft_compare.jpg")
    plt.show()
    plt.close()

    # ---------------------------------------------------------------------
    # Test main_fun utilities
    # ---------------------------------------------------------------------
    sample_input, sample_real = fn.load(os.path.join(base_path, "output/train/ellipse_001348.jpg"))
    print("Sample shapes:", np.shape(sample_input), np.shape(sample_real))

    # Save split images
    for idx, img in enumerate([sample_input, sample_real]):
        plt.imshow(img)
        plt.savefig(f"testing_image/sep{idx}.png")
        plt.show()

    # Test random jitter
    obs_jit, sky_jit = fn.random_jitter(sample_input, sample_real)
    for idx, img in enumerate([obs_jit, sky_jit], start=2):
        plt.imshow(img)
        plt.savefig(f"testing_image/jit{idx}.png")
        plt.show()

    # Multiple jitter examples
    plt.figure(figsize=(6, 6))
    for i in range(4):
        jit_obs, jit_sky = fn.random_jitter(sample_input, sample_real)
        plt.subplot(2, 2, i + 1)
        plt.imshow(jit_sky)
        plt.axis("off")
        plt.savefig(f"testing_image/jit_sky{i}.png")
    plt.show()

    # Salt & pepper noise
    spimage = fn.Saltandpepper(sample_real, 0.005, display_img=True)
    plt.imshow(spimage)
    plt.savefig("testing_image/salt_pepper.png")
    plt.show()

    # Downsampling
    down_model = fn.downsample(3, 4)
    down_result = down_model(tf.expand_dims(tf.cast(sample_input, float), 0))
    print("Downsampling result shape:", down_result.shape)

    # Upsampling
    up_model = fn.upsample(3, 4, apply_dropout=True)
    up_result = up_model(down_result)
    print("Upsampling result shape:", up_result.shape)


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform Fourier analysis and preprocessing tests."
    )
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Base path containing data directories (e.g. base_npy/, ellip_npy/, train/).",
    )
    args = parser.parse_args()

    main(args.base_path)
