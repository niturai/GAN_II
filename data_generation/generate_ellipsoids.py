"""
Batch-generate ellipsoid fields and save previews + compressed arrays.

- Headless matplotlib backend (safe on servers)
- Zero-padded filenames
- Optional fast previews (plt.imsave) or slower previews with colorbar
- Arrays saved as NPZ with metadata (rad, inc, pa, sq, ds, N, lam)
"""

# --- Headless backend (must come before pyplot import) ---
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

# Adjust to your real module path
from utils.ellipsoid import Ellipsoid


# ---------------- Configuration ----------------
USE_COLORBAR = False         # True = slower w/ colorbar; False = fastest
SAVE_PREVIEWS = False        # False = skip images entirely
DPI = 300
CMAP = None                  # e.g., "inferno" or None for default
# ------------------------------------------------


def _save_preview(arr: np.ndarray, out_path: Path):
    """Save a quick preview image."""
    if not SAVE_PREVIEWS:
        return
    if USE_COLORBAR:
        fig, ax = plt.subplots()
        im = ax.imshow(arr, cmap=CMAP)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    else:
        # Fast path: no figure, no colorbar
        plt.imsave(out_path, arr, cmap=CMAP)


def _save_npz(arr: np.ndarray, out_path: Path, *, rad, inc, pa, sq, ds, N, lam):
    """Save array + parameters in a compressed NPZ (self-describing sample)."""
    np.savez_compressed(
        out_path,
        data=arr.astype(np.float32),  # change to float64 if you need full precision
        rad=float(rad), inc=float(inc), pa=float(pa), sq=float(sq),
        ds=float(ds), N=int(N), lam=float(lam),
    )


def main():

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

    # Output dirs
    image_dir = Path(os.path.join(base_path, "ellip_image"))
    npz_dir   = Path(os.path.join(base_path, "ellip_npz"))
    image_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)

    # Generator
    ep = Ellipsoid()

    # Fixed parameters
    N = 512
    ds = 1e-10
    lam = 1e-6
    sx, sy, x, y = ep.grids(ds, N, lam)  # 512 x 512 arrays

    # Sweeps
    rad = np.arange(3e-9, 1.6e-8, 1e-9)   # 3e-9, 5e-9, ..., 1.4e-8
    inc = np.arange(0, 2*np.pi, np.pi/8)  # 0, π/4, ..., 7π/4
    pa  = np.arange(0, 2*np.pi, np.pi/8)
    sq  = np.arange(0.5, 1.0, 0.1)

    pic = 0

    # Pass 1: special case (inc near 0 or π) with pa=0, sq=1
    iterable_1 = ((r, i) for r in rad for i in inc)
    iterable_1 = tqdm(list(iterable_1), desc="inc in {0, π}")
    for r, i in iterable_1:
        # Nudge inclinations near π/2, 3π/2 to avoid singularities
        if np.isclose((i % np.pi), 0.5*np.pi):
            i = i + 0.07

        if np.isclose(i % np.pi, 0.0):  # near 0 or π
            field = ep.ellip(sx, sy, r, i, 0.0, 1.0)

            preview_path = image_dir / f"ellipse_{pic:06d}.jpg"
            npz_path = npz_dir / f"ellipse_{pic:06d}.npz"

            _save_preview(field, preview_path)
            _save_npz(field, npz_path, rad=r, inc=i, pa=0.0, sq=1.0, ds=ds, N=N, lam=lam)
            pic += 1

    # Pass 2: full sweep over pa and sq (skip the special-case combo to avoid duplicates)
    iterable_2 = product(rad, inc, pa, sq)
    total2 = len(rad) * len(inc) * len(pa) * len(sq)
    iterable_2 = tqdm(iterable_2, total=total2, desc="full sweep")

    for r, i, p, s in iterable_2:
        # Nudge inclinations near π/2, 3π/2 to avoid singularities
        if np.isclose((i % np.pi), 0.5*np.pi):
            i = i + 0.07

        # Skip duplicate special case: inc in {0, π} with pa=0 and sq=1
        if np.isclose(i % np.pi, 0.0) and np.isclose(p % (2*np.pi), 0.0) and np.isclose(s, 1.0):
            continue

        field = ep.ellip(sx, sy, r, i, p, s)

        preview_path = image_dir / f"ellipse_{pic:06d}.jpg"
        npz_path = npz_dir / f"ellipse_{pic:06d}.npz"

        _save_preview(field, preview_path)
        _save_npz(field, npz_path, rad=r, inc=i, pa=p, sq=s, ds=ds, N=N, lam=lam)
        pic += 1


if __name__ == "__main__":
    main()
