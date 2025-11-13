import numpy as np

class mom():
      """
      This class objects define moments for different order.
      """

      def mom_def(self, img):
          """
          calculate the moment of an image from 0 to 3rd order
          
          Input :
          -------
          img : the file with shape (x, y)
          
          Output :
          --------
          tuple : Monopole, centroid, 2nd and 3rd order moment
          """
    
          height, width = img.shape
          x = np.arange(0, width)
          y = np.arange(0, height)
          x, y = np.meshgrid(x, y)
    
          # monopole
          M00 = np.sum(img)

          # The centroid
          mx, my = np.sum(x*img)/M00, np.sum(y*img)/M00       

          # The second-order central moments
          mu11 = np.sum((x-mx) * (y-my) * img)/M00
          mu20 = np.sum((x-mx)**2 * img)/M00
          mu02 = np.sum((y-my)**2 * img)/M00

          # The third order moment
          mu30 = np.sum((x-mx)**3 * img)/M00
          mu03 = np.sum((y-my)**3 * img)/M00
          mu21 = np.sum((x-mx)**2 * (y-my) * img)/M00
          mu12 = np.sum((x-mx) * (y-my)**2 * img)/M00

          return M00, mx, my, mu11, mu20, mu02, mu30, mu03, mu21, mu12
          
      def akar(self, mu11, mu20, mu02):
          """
          Calculate the shape of object in given image
          
          Input :
          -------
          mu11 : The 2nd order moment along x and y axis
          mu20 : The 2nd order moment along x axis
          mu02 : The 2nd order moment along y axis
          
          Output :
          --------
          tuple : Orientation, semi-major, semi-minor, eccentricity and area of object.
          """
    
          muplus = (mu20 + mu02)/2
          muminus = (mu20 - mu02)/2
    
          # the orientation
          alpha = np.arctan2(mu11, muminus)/2

          # the eigen-vectors
          delta = (mu11**2 + muminus**2)**(1/2)
          lam1, lam2 = muplus + delta, muplus - delta

          # the semi-major and semi-minor axis
          a, b = 2*lam1**(1/2), 2*lam2**(1/2)

          # the eccentricity
          e = (1-lam2/lam1)**(1/2)

          # the area of ellipse
          A = 4*np.pi*(mu20*mu02 - mu11**2)**(1/2)
          print('reconstructed parameter : alpha=%f, a=%f b=%f, e=%f, Area=%f' % (alpha, a, b, e, A))
    
          return alpha, a, b, e, A
          
      def ellip(self, img, mx, my, alpha, a, e):
          """
          The structure of an ellipse for given Orientation, semi-major, eccentricity.
          
          Input :
          -----
          img : It will define the size of output image (x, y)
          
          mx : The centroid of object along x 
          
          my : The centroid of object along y
          
          alpha : The orientation of object
          
          a : The semi-major axis
           
          e : The eccentricity
          
          Output :
          -------
          metric : 2-d metric of structure of object
          """
      
          height, width = img.shape
          x = np.arange(0, width)
          y = np.arange(0, height)
          x, y = np.meshgrid(x, y)
    
          xp,yp = x-mx,y-my
          cs,sn = np.cos(alpha),np.sin(alpha)
          xp,yp = cs*xp + sn*yp, -sn*xp + cs*yp
          w = 0*x
          w[xp**2 + yp**2/(1-e*e) < a*a] = 1

          return w
          
      def prakar(self, mx, my, alpha, a, b):
          """
          The shape and size of object
          
          Input :
          -------
          mx : The centroid of object along x 
          
          my : The centroid of object along y
          
          alpha : The orientation of object
          
          a : The semi-major axis
           
          b : The semi-minor axis
          
          Output :
          -------
          tuple : array of x and y
          """
          t = np.linspace(0, 2*np.pi, 501)
    
          X = a * np.cos(t)
          Y = b * np.sin(t)
    
          x = X * np.cos(alpha) - Y * np.sin(alpha) + mx
          y = X * np.sin(alpha) + Y * np.cos(alpha) + my
    
          return x, y
          
          
