"""
Generate telescope layout, baselines, and (u, v) plane coverage for intensity interferometry.

Steps:
1. Define observation times (Julian Dates).
2. Define telescope positions at Veritas Observatory.
3. Plot telescope arrangement and save to PNG.
4. Compute baselines and their evolution with Earth rotation.
5. Plot covered (u, v) plane and save.
6. Convert (u, v) coverage to binary mask (grayscale threshold + resize),
   and save as a NumPy array.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from pathlib import Path

from utils.aperture import Aperture
from utils.obstime import ObsTime
from utils.obspoint import ObsPoint


def main():
    # --- Initialize classes ---
    ap = Aperture()
    obst = ObsTime()
    obsp = ObsPoint()


    # --- Parase base path ---
    parser = argparse.ArgumentParser(
        description="Generate telescope layout and baseline coverage."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path.cwd() / "base",
        help="Base directory for saving results (default: ./base)",
    )
    output_dir = parser.parse_args().path
    os.makedirs(output_dir, exist_ok=True)

    # --- Observation time for Intensity Interferometry ---
    start = [2460311.20167]  # JD (Jan 1, 2024, 7pm)
    end = [2460311.70833]    # JD (Jan 2, 2024, 5am)
    step = obst.obslen(start, end, 0.0104167)  # ~15 min steps
    jd = obst.julday(start, end, step)

    # --- Observatory location (Veritas) ---
    la = obsp.raddeg(31, 40, 30)
    lo = obsp.raddeg(110, 57, 7)

    # --- Telescope positions ---
    x = np.linspace(-60, 150, 500)
    y = np.linspace(-60, 80, 500)
    Tname = ["T1", "T2", "T3", "T4"]
    Tpos = [135 - 15j, 40 - 50j, 30 + 60j, -40 + 10j]

    # Apply rotation factor
    ang = np.exp(0.1j)
    Tpos = [ang * pos for pos in Tpos]

    # Save telescope arrangement plot
    ap.telescope(Tpos, Tname, os.path.join(output_dir, "telescope.png"), x, y, radii=6)

    # --- Baseline name and vectors ---
    tel, base = ap.baseline(Tpos, Tname)
    x = np.real(base)
    y = np.imag(base)
    z = 1e-6  # wavelength in meters?

    # --- Source coordinates (Spica) ---
    r = obsp.radhr(13, 25, 11.579)
    de = obsp.raddeg(-11, 9, 40.75)

    # --- Baseline variation due to Earth rotation ---
    obsp.ps_para(lat=la, lon=lo, rac=r, dec=de)
    dist = []
    for i in range(len(x)):
        dist.append(obsp.rotbase(x[i], y[i], z, jd))

    distance = np.array(dist)
    xt = distance[:, 0]  # baselines in east direction
    yt = distance[:, 1]  # baselines in north direction
    zt = distance[:, 2]  # baseline in up

    # --- Plot covered (u, v) plane ---
    plt.close()
    plt.rcParams.update({
        "font.size": 15,
        "figure.figsize": [12, 8],
        "font.weight": "bold",
        "axes.labelweight": "bold",
    })

    for k in range(len(xt)):
        plt.plot(xt[k, :], yt[k, :], ".", color="black")

    plt.axis("off")
    plt.gca().set_aspect("equal")
    plt.savefig(os.path.join(output_dir, "baseline.png"), dpi=500)

    # --- Convert (u, v) coverage to binary mask ---
    base_img = plt.imread(os.path.join(output_dir, "baseline.png"))

    image_size = 128  # px
    mask = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    mask = np.where(mask < 1, 1, 0).astype(np.float32)
    mask = cv2.resize(mask, dsize=(image_size, image_size),
                      interpolation=cv2.INTER_AREA)
    mask = np.where(mask > 0, 1, 0)

    np.save(os.path.join(output_dir, "base.npy"), mask)


if __name__ == "__main__":
    main()
