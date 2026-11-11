import matplotlib.colors as mcolors
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import colorsys

class aper():
      """
      Creates an object for the arrangement of plane\masked circular aperture's telescopes on (x, y).
      Also, arrange the baseline's name and length from these telescopes.
      """
      
      def circ(self, x, y, rad):
          """
          Return a circular aperture of telescope
          
          Input :
          -----
          x : the x-coordinate on 2-D grids
          y : the y-coordinate on 2-D grids
          rad : the radius of aperture
          
          Output :
          ------
                A 2-D matrix of circular aperture.
          """
          f = 0*x
          f[x*x + y*y < rad**2] = 1
          return f

      def telescope(self, Tposi, Tname, color, fname):
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
          plt.close()
          plt.rcParams.update({'font.size': 12})
          plt.rcParams["figure.figsize"] = [6,6]
          plt.rcParams['axes.facecolor']='ivory'
          plt.rcParams["font.weight"] = "bold"
          plt.rcParams["axes.labelweight"] = "bold"
          for j in range(len(Tposi)):
              east = np.real(Tposi)
              north = np.imag(Tposi)
              tel = Tname[j]
              plt.annotate(tel,  xy = (east[j], north[j]), size=10, color='darkred', fontweight='bold')
          for i in range(len(e)):   
              plt.plot(e[i], n[i], '--', color=color[i])
          plt.xlabel('Along U axis')
          plt.ylabel('Along V axis')
          plt.axis('equal')
          plt.title("Position of Telescopes", fontweight='bold')
          
          plt.savefig(fname, dpi=100, bbox_inches='tight')
          plt.close()

      def baseline(self, Tposi):
          """
          Return the length of baselines for N number of telescopes.

          Parameters :
          ----------

          Tposi : list
                  List of int which defines the position of telescopes in (x, y) plane. Exa. [135 - 15j, 40 - 50j].

          Returns :
          -------
                  List of tupples. Each tupples are the length of baseline in (x, y) direction.
          """
          
          comb = combinations(Tposi, 2)
         
          base = []
          for i in list(comb):
              base.append(i[1] - i[0])
          
          return base 
          
      def basename(self, Tname):
           """
           Return the name of baselines for N number of telescopes.

           Parameters :
           ----------
           
           Tname : list
                  List of str which defines the name of telescope. Exa. ['T1', 'T2'].

           Returns :
           -------
                  List of tupples, which are the name of baseline.
           """
           T = combinations((Tname), 2)
           tel = []
           for i in list(T):
               tel.append(i)
           return tel
           
      def track(self, ut, vt, step, color, fname):
          """
          Tracking of baselines with time.
          
          Input :
          -----
          ut : array
               Baselines in u-direction
          vt : array
               Baselines in v-direction
          step : number of steps in observational time
          color : list of name of colors
          fname : string to save file name.
          """
          plt.close()
          plt.rcParams.update({'font.size': 12})
          plt.rcParams["figure.figsize"] = [6,6]
          plt.rcParams['axes.facecolor']='ivory'
          plt.rcParams["font.weight"] = "bold"
          plt.rcParams["axes.labelweight"] = "bold"
          pic = 0
          for j in range(step[0]):
              for k in range(len(ut)):
                  plt.plot(ut[k,:j], vt[k,:j], '.', linewidth = 6, color = color[k])
                  plt.plot(ut[k,j], vt[k,j], marker = 'o', markersize = 8, color = 'darkred')
                  plt.xlim(ut.min() - 10, ut.max() + 10)
                  plt.ylim(vt.min() - 10, vt.max() + 10)
                  plt.xlabel("U in East Direction (meter)")
                  plt.ylabel("V in North Direction (meter)")
                  plt.title("The Track of All Baselines with Time at One Night", fontweight='bold')
              plt.savefig(fname + str(pic) + '.png')
              pic += 1
              plt.clf()
              
      def colors(self):
          """
          Return the list of dark colors
          """
          css4_colors = mcolors.CSS4_COLORS                              # Get all CSS4 colors
          dark_colors = []                                               # Convert hex to RGB, then to HLS to check luminance
          for name, hex in css4_colors.items():
              r, g, b = mcolors.to_rgb(hex)
              h, l, s = colorsys.rgb_to_hls(r, g, b)
              if l < 0.5:                                                # Lightness threshold for dark color
                 dark_colors.append(name)
              
          colors = dark_colors
          return colors

