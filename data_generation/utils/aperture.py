from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np


class Aperture():
    """
    Create and visualize arrangements of circular / masked circular telescope
    apertures on an (x, y) grid, and compute baselines between telescopes.

    Notes
    -----
    - Telescope positions `Tposi` are complex numbers: East + 1j*North (meters).
    - The `telescope` method draws each aperture (circular or cosine-masked)
      and annotates telescope names, then draws baseline segments.
    - The `baseline` method returns:
        * `tel`: name pairs (tuples of str),
        * `base`: complex baseline vectors (ΔEast + 1j*ΔNorth).
      These are vectors (not scalar lengths).
    """

    # a circular telescope
    def circ(self, x: np.ndarray, y: np.ndarray, rad: float) -> np.ndarray:
        """
        Binary circular aperture.

        Parameters
        ----------
        x, y : np.ndarray
            Meshgrid coordinates (same shape).
        rad : float
            Radius of the circular aperture (same units as x/y).

        Returns
        -------
        np.ndarray
            Array of 0/1 with 1 inside the disk of radius `rad`.
        """
        f = 0 * x
        f[x * x + y * y < rad ** 2] = 1
        return f

    # a circular aperture masked with cosine-squared function
    def aper(self, x: np.ndarray, y: np.ndarray, rad: float, b: float, psi: float) -> np.ndarray:
        """
        Cosine-squared masked circular aperture.

        The aperture is a circular disk of radius `rad`, multiplied by cos^2 modulation
        along the rotated x'-axis, where the frame is rotated by `psi`.

        Parameters
        ----------
        x, y : np.ndarray
            Meshgrid coordinates (same shape).
        rad : float
            Radius of the circular aperture.
        b : float
            Stripe width parameter for the cosine-squared mask.
        psi : float
            Rotation angle in radians (counter-clockwise), applied to (x, y).

        Returns
        -------
        np.ndarray
            Masked aperture values in [0, 1].
        """
        cs, sn = np.cos(psi), np.sin(psi)
        # rotate coordinates by psi
        x, y = cs * x - sn * y, sn * x + cs * y
        return self.circ(x, y, rad) * np.cos(np.pi * x / b) ** 2

    def telescope(
        self,
        Tposi: list,
        Tname: list,
        fname: str,
        xcord: np.ndarray,
        ycord: np.ndarray,
        radii: float,
        width: float | None = None,
        orien: float | None = None,
    ) -> None:
        """
        Plot the plane or cosine-masked circular apertures on the (x, y) grid.

        All apertures use the same radius.

        Parameters
        ----------
        Tposi : list
            Telescope positions as complex numbers: East + 1j*North (meters).
        Tname : list
            Telescope names aligned with `Tposi`.
        fname : str
            Output image file path (e.g., 'apertures.png').
        xcord, ycord : np.ndarray
            1D coordinate arrays defining the plotting grid span for x and y.
        radii : float
            Radius of each telescope aperture (meters).
        width : float, optional
            Cosine-mask stripe width parameter; if provided, apertures are masked.
        orien : float, optional
            Mask orientation angle in radians; used when `width` is not None.

        Returns
        -------
        None
            Saves a figure to `fname`.
        """
        # all unordered telescope pairs (for baseline segments and vectors)
        comb = combinations(Tposi, 2)
        base = []
        for i in list(comb):
            base.append(i)

        # endpoints for plotting baseline segments
        e = np.real(base)
        n = np.imag(base)

        # plotting grid
        x, y = np.meshgrid(xcord, ycord)

        # global pyplot state as in original (kept to avoid behavior change)
        plt.close()
        plt.rcParams.update({'font.size': 14})
        plt.rcParams["figure.figsize"] = [12, 12]
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"

        # precompute telescope coordinates once
        east = np.real(Tposi)
        north = np.imag(Tposi)

        # draw each aperture and annotate its name
        for j in range(len(Tposi)):
            if width is None:
                w = self.circ(x - east[j], y - north[j], radii)
            else:
                # if width is provided, assume orien is provided by caller (kept as-is)
                w = self.aper(x - east[j], y - north[j], radii, width, orien)
            plt.contour(x, y, w, colors='darkgrey')

            tel = Tname[j]
            plt.annotate(
                tel,
                xy=(east[j], north[j]),
                size=10,
                color='black',
                fontweight='bold'
            )

        # draw dashed baseline segments between telescope pairs
        for i in range(len(e)):
            plt.plot(e[i], n[i], '--')

        plt.xlabel('East in meters')
        plt.ylabel('North in meters')
        plt.title("Position of Telescopes", fontweight='bold')
        plt.savefig(fname)

    def baseline(self, Tposi: list, Tname: list):
        """
        Return baseline name pairs and baseline vectors for N telescopes.

        Parameters
        ----------
        Tposi : list[complex]
            Telescope positions as complex numbers: East + 1j*North (meters).
        Tname : list[str]
            Telescope names aligned with `Tposi`.

        Returns
        -------
        tel : list[tuple[str, str]]
            Name pairs (baseline identifiers).
        base : list[complex]
            Complex baseline vectors (ΔEast + 1j*ΔNorth). These are vectors, not scalar lengths.
        """
        T = combinations((Tname), 2)
        comb = combinations(Tposi, 2)

        tel = []
        for i in list(T):
            tel.append(i)

        base = []
        for i in list(comb):
            base.append(i[1] - i[0])

        return tel, base
