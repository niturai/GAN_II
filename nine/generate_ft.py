import matplotlib.pyplot as plt
from scipy import fft
import numpy as np
import cv2
import os

output_dir = "../result/nine"                                 # Output directories
for folder in ["testing_image", "ft"]:
    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

base = np.load(f"{output_dir}/base_npy/base.npy")            # Load baseline data, shape = (128, 128)
train_image = np.load("../result/ellipse_npy/ellipse19.npy") # Load image data, shape = (512, 512)

def plot_sky_image(data, filename):                          # display sky image with angular scale

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data, origin="lower", interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, pad=0.04, shrink=0.8)
    cbar.set_label("Intensity", fontweight="bold")
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.set_aspect("equal")
    ax.set_axis_off()

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

plot_sky_image(train_image, "../result/nine/testing_image/ellipse19.png") # Original stellar image: 512 x 512
train_image = cv2.resize(train_image, dsize=(128, 128), interpolation=cv2.INTER_AREA) # Resize image to 128 x 128

if base.shape != train_image.shape:
    raise ValueError(
        f"Baseline mask shape {base.shape} does not match "
        f"image shape {train_image.shape}. Resize or regenerate "
        "the baseline mask to match the FFT grid."
    )

# display Fourier plane with baseline scale
def plot_fourier(data, filename, title, log_scale=False):
    if log_scale:
        data_plot = np.log10(np.maximum(data, 1e-12))  # Avoid log(0)
    else:
        data_plot = data

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data_plot, origin="lower", interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, pad=0.04, shrink=0.8)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(15)  # Increase colorbar value font size

    ax.set_title(title, fontweight="bold", fontsize=15)
    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(filename, dpi=100, bbox_inches="tight")
    plt.close(fig)


fourier = np.abs(fft.fftshift(fft.fft2(fft.ifftshift(train_image))))
fourier_norm = fourier / fourier.max()

plot_fourier(fourier_norm, "../result/nine/ft/ft.jpg", "Normalized Fourier Amplitude (NFA)")
plot_fourier(fourier_norm, "../result/nine/ft/ft_log.jpg", "Log10(Normalized Fourier Amplitude)", log_scale=True)

# Fourier amplitude sampled by baselines
output = fourier_norm * base

plot_fourier(output, "../result/nine/ft/ft_base.jpg", "Normalized Sampled Fourier Amplitude")
plot_fourier(output, "../result/nine/ft/ft_log_base.jpg", "Log10(Normalized Sampled Amplitude)", log_scale=True)

# Full Fourier amplitude in the background, baseline coverage overlaid
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(fourier_norm, origin="lower", cmap="viridis", interpolation="nearest") # Background: full normalized Fourier amplitude

covered = base > 0
y, x = np.where(covered) # Overlay: baseline-covered pixels
sc = ax.scatter(x, y, c=output[covered], cmap='cool', alpha=0.5, s=10, marker="o") # Use the sampled Fourier amplitude to color the covered points
cbar = fig.colorbar(sc, ax=ax, pad=0.04, shrink=0.81)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight("bold")
    label.set_fontsize(15)  # Increase colorbar value font size

ax.set_title("Covered NFA using Baselines", fontweight="bold", fontsize=15)
ax.set_aspect("equal")
ax.set_axis_off()
fig.tight_layout()
fig.savefig("../result/nine/ft/ft_covered.jpg", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Finished: full Fourier amplitude and baseline coverage saved.")


# Side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

im0 = axes[0].imshow(fourier_norm, origin="lower")
axes[0].set_title("Normalized Fourier Amplitude", fontweight="bold", fontsize=15)
axes[0].set_aspect("equal")
axes[0].set_axis_off()
cbar0 = fig.colorbar(im0, ax=axes[0], pad=0.04, shrink=0.8)
for label in cbar0.ax.get_yticklabels():
    label.set_fontweight("bold")
    label.set_fontsize(15)  # Increase colorbar value font size
    
im1 = axes[1].imshow(output, origin="lower")
axes[1].set_title("Normalized Sampled Fourier Amplitude", fontweight="bold", fontsize=15)
axes[1].set_aspect("equal")
axes[1].set_axis_off()
cbar1 = fig.colorbar(im1, ax=axes[1], pad=0.04, shrink=0.8)
for label in cbar1.ax.get_yticklabels():
    label.set_fontweight("bold")

fig.tight_layout()
fig.savefig("../result/nine/ft/ft_Complete.jpg", dpi=100, bbox_inches="tight")
plt.close(fig)

print("Finished: all figures saved with image scales.")
