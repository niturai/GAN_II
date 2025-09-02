import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class Ellipsoid:
    """
    Module for generating and visualizing ellipsoids on spatial grids.

    This module defines the `Ellipsoid` class, which provides utilities
    for:
    - Creating spatial grids on the sky and observational plane,
    - Generating ellipsoid shapes with given parameters,
    - Visualizing the results as contour plots.
    """
    # Define mas
    MAS = 1e-3/(180/np.pi*3600)

    def grids(self, ds: float, N: int, lam: float):
        """
        Create spatial grids on the sky and the observational plane.

        Parameters
        ----------
        ds : float
            Grid spacing size.
        N : int
            Number of grid points along one axis.
        lam : float
            Observational wavelength in meters.

        Returns
        -------
        tuple of np.ndarray
            (sx, sy) : 2D grids on the sky.
            (x, y)   : 2D grids on the observational plane.
        """

        # Sky plane grid
        sx = (np.arange(N) - N // 2) * ds
        sx, sy = np.meshgrid(sx, sx)

        # Observational plane grid
        dx = lam / (N * ds)
        x = (np.arange(N) - N // 2) * dx
        x, y = np.meshgrid(x, x)

        return sx, sy, x, y

    def cen(self, f: np.ndarray, zoom: int):
        """
        Extract the central portion of an array with a given zoom factor.

        Parameters
        ----------
        f : np.ndarray
            Input 2D array.
        zoom : int
            Zoom factor (power of 2 is preferred).

        Returns
        -------
        np.ndarray
            Central cropped portion of `f`.
        """
        N = f.shape[0]
        M = N // (zoom * 2)
        return f[N // 2 - M:N // 2 + M, N // 2 - M:N // 2 + M]

    def draw(self, xc, yc, f, zoom, where, cmap='Greys_r', ceil=None, fceil=None, title=None):
        """
        Draw contour maps of data (on sky or ground).

        Parameters
        ----------
        xc, yc : np.ndarray
            Coordinate grids.
        f : np.ndarray
            Data to plot.
        zoom : int
            Zoom factor (power of 2 preferred).
        where : str
            'sky' or 'ground' (defines units and scaling).
        cmap : str, optional
            Colormap (default 'Greys_r').
        ceil : float, optional
            Upper bound for normalization (forces positive scale).
        fceil : float, optional
            Symmetric bound for positive and negative values.
        title : str, optional
            Title for the plot.
        """
        f = self.cen(f, zoom)
        plt.clf()

        fmax, fmin = f.max(), f.min()

        if where == 'sky':
            # Convert to milli-arcseconds (mas)
            sx, sy = self.cen(xc, zoom) / self.MAS, self.cen(yc, zoom) / self.MAS
            if ceil:
                fmin, fmax = 0, max(ceil, f.max())
            levs = np.linspace(fmin, fmax, 40)
            cs = plt.contourf(sx, sy, f, levs, cmap=cmap)
            plt.xlabel('mas')

        elif where == 'ground':
            # Rescale depending on magnitude of coordinates
            if xc[-1, -1] > 3e4:
                x, y = 1e-3 * self.cen(xc, zoom), 1e-3 * self.cen(yc, zoom)
                plt.xlabel('kilometres')
            elif xc[-1, -1] < 1:
                x, y = 1e3 * self.cen(xc, zoom), 1e3 * self.cen(yc, zoom)
                plt.xlabel('millimetres')
            else:
                x, y = self.cen(xc, zoom), self.cen(yc, zoom)
                plt.xlabel('metres')

            # Normalize data
            if ceil:
                fmin, fmax = 0, max(ceil, f.max())
                levs = np.linspace(fmin, fmax, 20)
            elif fceil:
                fmax = max(fceil, f.max())
                fmin = -fmax
                levs = np.linspace(fmin, fmax, 80)
            else:
                fmin, fmax = 0, f.max()
                levs = np.linspace(fmin, fmax, 20)

            cs = plt.contourf(
                x, y, f, levs,
                norm=colors.Normalize(vmin=fmin, vmax=fmax),
                cmap=cmap
            )

        # Colorbar formatting
        if fmax > 10:
            fms = '%i'
        else:
            lgf = np.log10(fmax)
            ip = int(-lgf) + 2
            if lgf < -5:
                fms = '%7.1e'
            else:
                fms = f'%.{ip}f'

        # plt.colorbar(cs, format=fms)   # Uncomment if colorbar is desired
        if title:
            plt.title(title)
        plt.gca().set_aspect('equal')

    def ellip(self, sx, sy, rad, inc, pa, sq):
        """
        Generate ellipsoids of different size, angle, and shape.

        Parameters
        ----------
        sx, sy : np.ndarray
            Grid coordinates.
        rad : float
            Size of the ellipsoid.
        inc : float
            Inclination of the ellipsoid [0, 2π].
        pa : float
            Position angle (rotation around z-axis) [0, 2π].
        sq : float
            Thickness of the ellipsoid along z-axis.

        Returns
        -------
        np.ndarray
            2D array representing the ellipsoid.
        """
        cs, sn = np.cos(pa), np.sin(pa)

        # Rotate in xy plane around z-axis
        x, y = cs * sx + sn * sy, -sn * sx + cs * sy

        cosI, sinI = np.cos(inc), np.sin(inc)

        Tv = np.zeros_like(x)  # empty grid

        # Loop through angles to fill ellipsoid
        for rs in np.linspace(0, np.pi, 101):
            cs, sn = np.cos(rs), np.sin(rs)
            z = sinI * cs * sq * rad
            Tv[x**2 + ((y - z) / cosI) ** 2 < (sn * rad) ** 2] = (4 + abs(cs)) / 5

            # Optional visualization
            if np.min(Tv) < np.max(Tv):
                self.draw(sx, sy, Tv, 2, 'sky', ceil=1, cmap='inferno')

        return Tv
