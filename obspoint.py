import numpy as np

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

      def ps_para(self, lat, lon, rac, dec):
          """
           Set the parameters for position of telescope and binary source.
          
           Parameters :
           ----------

           lat : float
                 Latitude of observatory in radian.
           lon : float
                 Longitude of observatory in radian.
           rac : float
                 Right Ascension of source in radian
           dec : float
                 declination of source in radian

           Returns :
           -------
                   None.
           """
          self.lat = lat
          self.lon = lon
          self.rac = rac
          self.dec = dec

      # definition of the change in the baseline due to earth rotation
      def rotx(self, x, y, z, a):
          """
          """
          cs,sn = np.cos(a), np.sin(a)
          return x, cs*y - sn*z, sn*y + cs*z

      def roty(self, x, y, z, a):
          """
          """
          cs,sn = np.cos(a), np.sin(a)
          return cs*x + sn*z, y, -sn*x + cs*z

      def rotbase(self, dx, dy, dz, jd):
          """
          Variational baseline according to julian days (rotation of earth).
          
          Parameters :
          ----------
          
          dx : float
               Baseline in X-direction.
          dy : float
               Baseline in Y-direction.
          dz : float
               Baseline in Z-direction.
          jd : array
               All observational julian days.
          
          Returns :
          -------
          dx, dy, dz : array
                      These arrays are variational baselines along X, Y and Z direction.
          """
          # Define Hour Angle                                         
          gsid = 18.697374558 + 24.06570982441908*(jd - 2451545)   
          sid = (gsid % 24)*np.pi/12 + self.lon                     # in 1 hour 15 degree of rotation of earth
          ha = sid - self.rac
          dx,dy,dz = self.rotx(dx,dy,dz,-self.lat)
          dx,dy,dz = self.roty(dx,dy,dz,ha)
          dx,dy,dz = self.rotx(dx,dy,dz,self.dec)
          return dx, dy, dz


