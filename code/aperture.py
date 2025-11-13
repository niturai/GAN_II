from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np

class aper():
      """
      Creates an object for the arrangement of plane\masked circular aperture's telescopes on (x, y).
      Also, arrange the baseline's name and length from these telescopes.
      """
      # a circular telescope
      def circ(self, x, y, rad):
          """
          """
          f = 0*x
          f[x*x + y*y < rad**2] = 1
          return f

      # a circular aperture is masked with cosine square function
      def aper(self, x, y, rad, b, psi):
          """
          """
          cs, sn = np.cos(psi), np.sin(psi)
          x, y = cs*x - sn*y, sn*x + cs*y
          return self.circ(x, y, rad) * np.cos(np.pi*x/b)**2

      def telescope(self, Tposi, Tname, fname, xcord, ycord, radii, width=None, orien=None):
          """
          Plot the plane or masked aperture (follow cosine square) telescopes on the (x, y) coordinate.

          All apertures are the same size in diameter.

          Parameters :
          ----------

          Tposi : list
                  List of int which defines the position of telescopes in (x, y) plane. Exa. [135 - 15j, 40 - 50j].
          Tname : list
                  List of str which defines the name of the telescope. Exa. ['T1', 'T2'].
          fname : str
                  The name of the output file.
          xcord : ndarray
                  N number of samples evenly spaced in the x-direction in an interval over which the telescopes are placed.
          ycord : ndarray
                  N number of samples evenly spaced in the y-direction in an interval over which the telescopes are placed.
          radii : int or float
                  The radius of telescopes.
          width : float
                  The width of masking strips.
          orien : float
                  The orientation of masking strips in radian.

          Returns :
          -------
                  Two-dimensional arrangement of telescopes in .png format.
          """
          comb = combinations(Tposi, 2)
          base = []
          for i in list(comb):
              base.append(i)
          e = np.real(base)
          n = np.imag(base)
          x, y = np.meshgrid(xcord, ycord)
          plt.close()
          plt.rcParams.update({'font.size': 14})
          plt.rcParams["figure.figsize"] = [12,12]
          plt.rcParams["font.weight"] = "bold"
          plt.rcParams["axes.labelweight"] = "bold"
          for j in range(len(Tposi)):
              east = np.real(Tposi)
              north = np.imag(Tposi)
              if width == None:
                 w = self.circ(x-east[j], y-north[j], radii)
              else:
                 w = self.aper(x-east[j], y-north[j], radii, width, orien)
              plt.contour(x, y, w, colors='darkgrey')
              tel = Tname[j]
              plt.annotate(tel,  xy = (east[j], north[j]), size=10, color='black', fontweight='bold')
          for i in range(len(e)):   
              plt.plot(e[i], n[i], '--')
          plt.xlabel('East in meter')
          plt.ylabel('North in meter')
          plt.title("Position of Telescopes", fontweight='bold')
          plt.savefig(fname)
          

      def baseline(self, Tposi, Tname):
          """
          Return the name and length of baselines for N number of telescopes.

          Parameters :
          ----------

          Tposi : list
                  List of int which defines the position of telescopes in (x, y) plane. Exa. [135 - 15j, 40 - 50j].
          Tname : list
                  List of str which defines the name of telescope. Exa. ['T1', 'T2'].

          Returns :
          -------
          bname : list
                  List of tupples, which are the name of baseline.
          blen  : list
                  List of tupples. Each tupples are the length of baseline in (x, y) direction.
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


