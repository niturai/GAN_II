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
          
          
