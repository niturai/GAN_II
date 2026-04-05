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
          x = np.linspace(-64, 64, width)
          y = np.linspace(-64, 64, height)
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
          
      def mom_cal(self, imgG, imgP):
    
          n = imgG.shape[0]
    
          mxG, myG = [], []
          mxP, myP = [], []
    
          mu11G, mu20G, mu02G, mu30G, mu03G, mu21G, mu12G = [], [], [], [], [], [], []
          mu11P, mu20P, mu02P, mu30P, mu03P, mu21P, mu12P = [], [], [], [], [], [], []

          for i in range(n):
              M00g, mxg, myg, mu11g, mu20g, mu02g, mu30g, mu03g, mu21g, mu12g = self.mom_def(imgG[i])
              M00p, mxp, myp, mu11p, mu20p, mu02p, mu30p, mu03p, mu21p, mu12p = self.mom_def(imgP[i])

              mxG.append(mxg); myG.append(myg)
              mxP.append(mxp); myP.append(myp)

              mu11G.append(mu11g); mu20G.append(mu20g); mu02G.append(mu02g)
              mu30G.append(mu30g); mu03G.append(mu03g); mu21G.append(mu21g); mu12G.append(mu12g)

              mu11P.append(mu11p); mu20P.append(mu20p); mu02P.append(mu02p)
              mu30P.append(mu30p); mu03P.append(mu03p); mu21P.append(mu21p); mu12P.append(mu12p)
              
          mxG, myG = np.array(mxG), np.array(myG)
          mxP, myP = np.array(mxP), np.array(myP)

          mu11G, mu20G, mu02G = np.array(mu11G), np.array(mu20G), np.array(mu02G)
          mu11P, mu20P, mu02P = np.array(mu11P), np.array(mu20P), np.array(mu02P)

          mu30G, mu03G = np.array(mu30G), np.array(mu03G)
          mu30P, mu03P = np.array(mu30P), np.array(mu03P)

          mu21G, mu12G = np.array(mu21G), np.array(mu12G)
          mu21P, mu12P = np.array(mu21P), np.array(mu12P)
          
          # calculate the scatter in centroid
          Sc = n**(-1) * np.sqrt(np.sum((mxG - mxP)**2 + (myG - myP)**2))
          
          # calculate the scatter in central moment
          S11 = n**(-1) * np.sqrt(np.sum((mu11G - mu11P)**2))
          S20 = n**(-1) * np.sqrt(np.sum((mu20G - mu20P)**2))
          S02 = n**(-1) * np.sqrt(np.sum((mu02G - mu02P)**2))
          S12 = n**(-1) * np.sqrt(np.sum((mu12G - mu12P)**2))
          S21 = n**(-1) * np.sqrt(np.sum((mu21G - mu21P)**2))
          S30 = n**(-1) * np.sqrt(np.sum((mu30G - mu30P)**2))
          S03 = n**(-1) * np.sqrt(np.sum((mu03G - mu03P)**2))
          
          return Sc, S11, S20, S02, S12, S21, S30, S03
