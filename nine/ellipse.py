import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

mas = 1e-3/(180/np.pi*3600)

class ellipsoid():
      """
      This class objects creats ellipsoids with different size and shape.
      """

      def grids(self, ds, N, lam):
          """
          create grids on sky and observational plane
          
          Input :
          -------
          ds : the grid's size
          N : Number of grids
          lam : observational wavelength in meter
          
          Output :
          --------
          two dimensional grids on sky and observational plane    
          """
          
          # on sky
          sx = (np.arange(N) - N//2) * ds
          sx, sy = np.meshgrid(sx, sx)
    
          # on observational plane
          dx = lam/(N*ds)
          x = (np.arange(N) - N//2) * dx
          x, y = np.meshgrid(x, x)
          
          return sx, sy, x, y

      def cen(self, f, zoom):          
          N = f.shape[0]
          M = N // (zoom * 2)
          return f[N // 2 - M:N // 2 + M, N // 2 - M:N // 2 + M]
          
      def draw(self, xc, yc, f, zoom, where, cmap='Greys_r', ceil=None, fceil=None, title=None):
          """
          Draws contour map of f (on sky or ground, directed)
          zoom factor (power of 2 preferred)
          """
          
          f = self.cen(f, zoom)
          plt.clf()
          #plt.tight_layout()
          fmax = f.max()
          fmin = f.min()
          if where == 'sky':
              sx, sy = self.cen(xc, zoom) / mas, self.cen(yc, zoom) / mas
              if ceil:
                  fmin, fmax = 0, max(ceil, f.max())
              levs = np.linspace(fmin, fmax, 40)
              cs = plt.contourf(sx, sy, f, levs, cmap=cmap)
              plt.xlabel('mas')
          if where == 'ground':
              if xc[-1, -1] > 3e4:
                  x, y = 1e-3 * self.cen(xc, zoom), 1e-3 * self.cen(yc, zoom)
                  plt.xlabel('kilometres')
              elif xc[-1, -1] < 1:
                  x, y = 1e3 * self.cen(xc, zoom), 1e3 * self.cen(yc, zoom)
                  plt.xlabel('millimetres')
              else:
                  x, y = self.cen(xc, zoom), self.cen(yc, zoom)
                  plt.xlabel('metres')
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
              cs = plt.contourf(x, y, f, levs, norm=colors.Normalize(vmin=fmin, vmax=fmax), cmap=cmap)
          if fmax > 10:
              fms = '%i'
          else:
              lgf = np.log10(fmax)
              ip = int(-lgf) + 2
              if lgf < -5:
                  fms = '%7.1e'
              else:
                  fms = '%' + '.%i' % ip + 'f'
          #plt.colorbar(cs) #,format=fms)
          #plt.axis('off')
          if title:
             plt.title(title)
          plt.gca().set_aspect('equal')

      def ellip(self, sx, sy, rad, inc, pa, sq):
          """
          It generates ellipsoids of different size, angle and shape, and saves the images as .jpg images
          
          Parameters:
          -----------
          rad: size of the ellipsoid [3e-9 and 1.5e-8]
          inc: Inclination of the ellipsoid [0, 2pi]
          pa: Position angle; Rotation around x/y-axis [0, 2pi]
          sq: Axis Ratio; Thickness of the ellipsoid along z-axis [0.6, 1]
          
          Output: 
          -------
          .jpg image & .npx array
          """
          
          cs, sn = np.cos(pa), np.sin(pa)
          x, y = cs*sx + sn*sy, -sn*sx + cs*sy                            # rotation in xy plane around z-axis with an angle pa
          cosI, sinI = np.cos(inc), np.sin(inc)                           # inclination of ellipsoids in 2-D plane with respect to x-axis
          Tv = 0*x                                                        # all the grids are none i.e. no ellipsoids
          for rs in np.linspace(0, np.pi, 101):                           # for each position in ellipsoids
              cs, sn = np.cos(rs), np.sin(rs)
              z = sinI * cs * sq * rad                                    # the value grids along the z axis
              Tv[x**2 + ((y-z)/cosI)**2 < (sn*rad)**2] = (4 + abs(cs))/5  # fill the grids of ellipsoids
              if np.min(Tv) < np.max(Tv):
                  self.draw(sx,sy,Tv,2,'sky',ceil=1,cmap='inferno')       # plot the ellipsoids on 2-D plane
          return Tv
 
         
