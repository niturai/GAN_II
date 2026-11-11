import matplotlib.pyplot as plt
from ellipse import ellipsoid
import numpy as np
import os

ep = ellipsoid()

# Grid parameters
ds = 1e-10           # Angular pixel spacing in radians
N = 512              # Number of grids
lam = 1e-6           # Wavelength in metres

sx, sy, x, y = ep.grids(ds, N, lam)

rad_to_mas = (180 / np.pi) * 3600 * 1000 # multiply it to Convert radians to milliarcseconds
image_scale = ds * rad_to_mas
fov = N * image_scale
print(f"Image scale = {image_scale:.6f} mas/pixel")
print(f"Field of view = {fov:.4f} mas")

# Ellipse parameters
rad = np.arange(3e-9, 1.6e-8, 2e-10) # in Radian
inc = np.arange(0, 2*np.pi, np.pi/4)
pa = np.arange(0, 2*np.pi, np.pi/4)
sq = np.arange(0.5, 1, 0.1)

image_dir = '../result/ellipse_image'
npy_dir = '../result/ellipse_npy'
os.makedirs(image_dir, exist_ok=True)
os.makedirs(npy_dir, exist_ok=True)
pic = 0
extent = [
    sx.min() * rad_to_mas - image_scale/2,
    sx.max() * rad_to_mas + image_scale/2,
    sy.min() * rad_to_mas - image_scale/2,
    sy.max() * rad_to_mas + image_scale/2
]                                               # Image extent in milliarcseconds

def save_ellipse(ellipse, pic):
    fig, ax = plt.subplots(figsize=(6, 6))
 
    im = ax.imshow(ellipse, origin='lower', extent=extent, interpolation='nearest', aspect='equal') # Display ellipse
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)    # Add colorbar
    cbar.set_label('Intensity', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    for label in cbar.ax.get_yticklabels():  # Make colorbar tick values bold
        label.set_fontweight('bold')

    ax.set_xlabel('Right Ascension (mas)', fontsize=12, fontweight='bold') # Axis labels
    ax.set_ylabel('Declination (mas)', fontsize=12, fontweight='bold')

    ax.tick_params(axis='both', labelsize=10)                              # Make axis tick values bold
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    fig.tight_layout()
    fig.savefig(os.path.join(image_dir, f'ellipse{pic}.jpg'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    np.save(os.path.join(npy_dir, f'ellipse{pic}.npy'), ellipse)           # Save original numerical array unchanged

for r in rad:
    for i in inc:
        if np.isclose(i, np.pi/2) or np.isclose(i, 3*np.pi/2):             # Avoid singularities near edge-on orientations
            i += 0.07
        if np.isclose(i, 0) or np.isclose(i, np.pi):
            ellipse = ep.ellip(sx, sy, r, i, 0, 1)
            save_ellipse(ellipse, pic)
            pic += 1

        else:
            for p in pa:
                for s in sq:
                    ellipse = ep.ellip(sx, sy, r, i, p, s)
                    save_ellipse(ellipse, pic)
                    pic += 1

print(f"Finished. Saved {pic} images.")
