from aperture import aper
import numpy as np

apr = aper()

class vary_base():
      """
      Creates an object that reads the position of the observatory and source in radian.
      It return the variational position of baselines and the observational grids point.
      """
      def raddeg(self, d, m, s):
          """
          Convert the given degree to radian.
          
          Parameters :
          ----------

          d : int
              In degrees
          m : int
              In minutes
          s : float
              In seconds
            
          Returns :
          -------
                  Return degree to radian
          """
          return (d + m/60 + s/3600) * np.pi/180

      def radhr(self, h, m, s):
          """
          Convert the given hours to radian.
          
          Parameters :
          ----------

          h : int
              In hour
          m : int
              In minutes
          s : float
              In seconds
            
          Returns :
          -------
                  Return hours to radian
          """
          return (h + m/60 + s/3600) * np.pi/12
          
      def deg(self, d, m, s):
          """
          convert degree, minutes and second in complete degree
          
          d : int
              In degrees
          m : int
              In minutes
          s : float
              In seconds
            
          Returns :
          -------
                  Return degree
          """
          return d + m/60 + s/3600

      def ps_para(self, rac, dec, R, lam):
          """
           Set the parameters for binary source and observation.
          
           Parameters :
           ----------

           rac : float
                 Right Ascension of source in radian
           dec : float
                 declination of source in radian
           R : float
               The radius of Earth in meter
           lam : float
                 The observational wavelength in meter

           Returns :
           -------
                   None.
           """
          self.rac = rac
          self.dec = dec
          self.R = R
          self.lam = lam

      def rotx(self, x, y, z, a):
          """
          Rotational matrix along x-axis.
          """
          cs,sn = np.cos(a), np.sin(a)
          return x, cs*y - sn*z, sn*y + cs*z

      def roty(self, x, y, z, a):
          """
          Rotational matrix along y-axis.
          """
          cs,sn = np.cos(a), np.sin(a)
          return cs*x + sn*z, y, -sn*x + cs*z

      def rotbase(self, lat, lon, jd):
          """
          Variational baseline according to julian days (rotation of earth).
          
          Parameters :
          ----------
          
          lat : array
                the latitude of each telescopes
          lon : array
                the longitude of each telescopes
          jd : array
               All observational julian days.
          
          Returns :
          -------
          u, v, w : array
                    The variational baselines along U, V and W direction.
          """ 
       
          E = self.R * np.cos(lat) * np.sin(lon)
          N = self.R * np.sin(lat)
          up = self.R * np.cos(lat) * np.cos(lon)
          
          Tpos = E + N * 1j
          
          Tpos = []
          for i in range(len(E)):
              Tpos.append(np.array([E[i], N[i], up[i]]))
          
          base = apr.baseline(Tpos)
          dpx = [x[0] for x in base]
          dpy = [x[1] for x in base]
          dpz = [x[2] for x in base]
          
          ut = []
          vt = []
          wt = []
          for i in range(len(dpx)):                                        
              gsid = 18.697374558 + 24.06570982441908*(jd - 2451545)   
              sid = (gsid % 24)*np.pi/12                            # in 1 hour 15 degree of rotation of earth
              ha = sid - self.rac                                   # Define Hour Angle
          
              dx, dy, dz = self.roty(dpx[i], dpy[i], dpz[i], ha)
              dx, dy, dz = self.rotx(dx, dy, dz, self.dec)
              ut.append(dx)
              vt.append(dy)
              wt.append(dz)              
              
          return np.array(ut), np.array(vt), np.array(wt)

      def grids(self, xcord, ycord, radii, steps):
          """
          Two dimensional grids for each observatinal interval.
          
          Grids are for given baseline and aperture.
          
          Parameters :
          ----------
          
          xcord : array
                  The baseline position, along x-direction for each observational interval.
          ycord : array
                  The baseline position, along y-direction for each observational interval.
          radii : float
                  The radius of aperture of telescopes.
          steps : int
                  Defines the grid's size.
              
          Returns :
          -------
          gX : N x N 
               For the baseline along x-direction.
          gY : N x N 
               For the baseline along y-direction.
          wX : N x N 
               For the aperture along x-direction.
          wY : N x N 
               For the aperture along y-direction.
          """
          N = steps
          xup, xdn = np.max(xcord), np.min(xcord)                           
          xmid, xh = (xup + xdn)/2, (xup - xdn)/2                   
          yup, ydn = np.max(ycord), np.min(ycord)
          ymid, yh = (yup + ydn)/2, (yup - ydn)/2                   
          hr = max(xh,yh)                                           
          r = np.linspace(-hr-2*radii, hr+2*radii, N)                   
          wX, wY = np.meshgrid(r,r)                                 
          gX, gY = xmid + wX, ymid + wY                             
          return gX, gY, wX, wY
          

