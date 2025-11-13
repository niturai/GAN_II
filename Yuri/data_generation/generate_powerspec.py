"""
Script to generate datasets for pix2pix training from stellar source images.

Steps:
1. Load baseline mask (sparse sampling pattern).
2. For each ellipsoid .npz file:
   - Extract stored array.
   - Add salt & pepper noise.
   - Compute FFT power spectrum.
   - Apply baseline mask.
   - Normalize and combine with ground truth image.
3. Split results into train, val, and test sets by percentage.
4. Save combined images to disk as JPEGs.
"""

import os
import random
import numpy as np
import cv2
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import fft

from utils.image_noise import ImageNoise
from utils.image_merge import ImageMerge


def main():
    # --- Config ---
    image_size = 128            # px
    alpha = 0.005               # Salt and Pepper noise probability
    save_images = True
    split_ratio = (0.7, 0.15, 0.15)  # train, val, test fractions

    imnoise = ImageNoise()
    imadd = ImageMerge()

    # Parse base path
    parser = argparse.ArgumentParser(
        description="Generate ellipsoid fields and save arrays + previews."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path.cwd() / "outputs",
        help="Base directory for saving results"
    )
    base_path = parser.parse_args().path
    path_in = os.path.join(base_path, "ellip_npz/")      # directory with input .npz ellipsoid images
    out_dir = os.path.join(base_path, "output")

    # --- Output dirs ---
    for d in ["train", "val", "test"]:
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    # --- Load baseline mask ---
    base = np.load(os.path.join(base_path, "base.npy"))

    # --- Collect and shuffle input files ---
    files = [f for f in os.listdir(path_in) if f.endswith(".npz")]
    random.shuffle(files)

    total = len(files)
    n_train = int(total * split_ratio[0])
    n_val = int(total * split_ratio[1])
    n_test = total - n_train - n_val

    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:]
    }

    print(f"Total files: {total} → train={n_train}, val={n_val}, test={n_test}")

    counter = 0
    for split_name, split_files in splits.items():
        for filename in split_files:
            f = os.path.join(path_in, filename)

            # --- Load .npz image ---
            with np.load(f) as data:
                image_original = data["data"]  # consistent with generator

            img_org = image_original.copy()

            # Resize ground truth
            ground_truth = cv2.resize(
                img_org, dsize=(image_size, image_size),
                interpolation=cv2.INTER_AREA
            )

            # Add noise
            noisy = imnoise.sap_noise(img_org, alpha)
            img_ed = cv2.resize(noisy, dsize=(image_size, image_size),
                                interpolation=cv2.INTER_AREA)
            img_ed = img_ed - np.mean(img_ed)

            # FFT power spectrum
            img_fft = fft.fftshift(fft.fft2(fft.fftshift(img_ed)))
            fft_argument = np.abs(img_fft)

            # Apply baseline mask and normalize
            fft_argument = np.multiply(fft_argument, base)
            img_fft_norm = fft_argument / np.max(fft_argument)

            # Convert to uint8
            img_fft_norm = cv2.convertScaleAbs(img_fft_norm, alpha=255.0)
            ground_truth = cv2.convertScaleAbs(ground_truth, alpha=255.0)

            # Merge ground truth and FFT image
            combined_image = imadd.add_image(ground_truth, img_fft_norm)

            # Ensure correct dtype
            if combined_image.dtype != np.uint8:
                combined_image = cv2.convertScaleAbs(combined_image)

            # --- Save ---
            image_name = filename[:-4]  # strip .npz
            out_path = os.path.join(out_dir, split_name, f"{image_name}.jpg")

            if save_images:
                cv2.imwrite(out_path, combined_image)
            else:
                # Debug mode
                plt.imshow(ground_truth, cmap="gray")
                plt.show()
                plt.imshow(img_fft_norm, cmap="gray")
                plt.show()
                return  # exit after showing a couple

            counter += 1
            if counter % 50 == 0:
                print(f"{counter} images processed...")

    print(f"Finished: {counter} images saved.")


if __name__ == "__main__":
    main()
