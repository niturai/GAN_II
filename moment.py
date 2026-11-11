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
    
          M00 = np.sum(img)                                 # monopole

          mx, my = np.sum(x*img)/M00, np.sum(y*img)/M00     # The centroid    

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

          mxG, myG, mxP, myP = [], [], [], []

          mu11G, mu20G, mu02G = [], [], []
          mu30G, mu03G, mu21G, mu12G = [], [], [], []

          mu11P, mu20P, mu02P = [], [], []
          mu30P, mu03P, mu21P, mu12P = [], [], [], []

          for i in range(n):
              (M00g, mxg, myg, mu11g, mu20g, mu02g, mu30g, mu03g, mu21g, mu12g) = self.mom_def(imgG[i])
              (M00p, mxp, myp, mu11p, mu20p, mu02p, mu30p, mu03p, mu21p, mu12p) = self.mom_def(imgP[i])

              mxG.append(mxg)
              myG.append(myg)
              mxP.append(mxp)
              myP.append(myp)

              mu11G.append(mu11g)
              mu20G.append(mu20g)
              mu02G.append(mu02g)
              mu30G.append(mu30g)
              mu03G.append(mu03g)
              mu21G.append(mu21g)
              mu12G.append(mu12g)

              mu11P.append(mu11p)
              mu20P.append(mu20p)
              mu02P.append(mu02p)
              mu30P.append(mu30p)
              mu03P.append(mu03p)
              mu21P.append(mu21p)
              mu12P.append(mu12p)
 
          # Convert lists to NumPy arrays
          mxG, myG = np.array(mxG), np.array(myG)
          mxP, myP = np.array(mxP), np.array(myP)

          moment_pairs = {
              "mu11": (np.array(mu11G), np.array(mu11P)),
              "mu20": (np.array(mu20G), np.array(mu20P)),
              "mu02": (np.array(mu02G), np.array(mu02P)),
              "mu12": (np.array(mu12G), np.array(mu12P)),
              "mu21": (np.array(mu21G), np.array(mu21P)),
              "mu30": (np.array(mu30G), np.array(mu30P)),
              "mu03": (np.array(mu03G), np.array(mu03P)),
          }

          # 1. Existing scatter measures
          Sc = np.sqrt(np.sum((mxG - mxP)**2 + (myG - myP)**2)) / n
          scatter = {}
          for name, (gt, pred) in moment_pairs.items():
              scatter[name] = np.sqrt(np.sum((gt - pred)**2)) / n

          S11 = scatter["mu11"]
          S20 = scatter["mu20"]
          S02 = scatter["mu02"]
          S12 = scatter["mu12"]
          S21 = scatter["mu21"]
          S30 = scatter["mu30"]
          S03 = scatter["mu03"]
          
          return (Sc, S11, S20, S02, S12, S21, S30, S03)
          
      def asym(self, img):
          img180 = img[::-1, ::-1]             # Rotate by 180 degrees          
          resi = img - img180                  # Residual
 
          M00 = np.sum(img)                    # Monopole (total flux)
          A = np.sum(np.abs(resi)) / M00       # Asymmetry statistic

          return A
          
      def shape_parameters(self, img):
          """
          Calculate geometric parameters from the second-order
          central moments of an image.

          Returns:
              alpha : Position angle (degrees)
              a     : Semi-major axis
              b     : Semi-minor axis
              e     : Eccentricity
          """

          M00, mx, my, mu11, mu20, mu02, mu30, mu03, mu21, mu12 = self.mom_def(img)

          mp = (mu20 + mu02) / 2.0
          delta = np.sqrt(((mu20 - mu02) / 2.0)**2 + mu11**2)

          lam1 = mp + delta
          lam2 = mp - delta

          # Avoid small negative values due to numerical precision
          lam1 = max(lam1, 0.0)
          lam2 = max(lam2, 0.0)

          # Position angle, accounting for the 180-degree ambiguity
          alpha = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
          alpha = np.degrees(alpha) % 180.0

          # Semi-major and semi-minor axes
          a = 2.0 * np.sqrt(lam1)
          b = 2.0 * np.sqrt(lam2)

          # Eccentricity
          if lam1 > 0:
              e = np.sqrt(max(0.0, 1.0 - lam2 / lam1))
          else:
              e = np.nan

          return alpha, a, b, e


      def parameter_errors(self, imgG, imgP):
          """
          Calculate geometric parameters and their reconstruction
          errors for a set of ground-truth and predicted images.

          Parameters
          ----------
          imgG : Ground-truth images, shape (N, H, W)
          imgP : Reconstructed images, shape (N, H, W)

          Returns
          -------
          results : Dictionary containing per-image errors and
                    summary statistics.
          """

          imgG = np.asarray(imgG)
          imgP = np.asarray(imgP)

          if imgG.shape != imgP.shape:
              raise ValueError("Ground-truth and predicted image arrays must have the same shape.")

          n = imgG.shape[0]
          if n < 2:
              raise ValueError("At least two test images are required to calculate standard deviation.")

          parameters = ["PA", "a", "b", "e"]
          gt_values = {key: [] for key in parameters}
          pred_values = {key: [] for key in parameters}

          # Calculate parameters for every image
          for i in range(n):
              alpha_g, a_g, b_g, e_g = self.shape_parameters(imgG[i])
              alpha_p, a_p, b_p, e_p = self.shape_parameters(imgP[i])

              values_g = [alpha_g, a_g, b_g, e_g]
              values_p = [alpha_p, a_p, b_p, e_p]

              for key, vg, vp in zip(parameters, values_g, values_p):
                  gt_values[key].append(vg)
                  pred_values[key].append(vp)

          # Convert to NumPy arrays
          for key in parameters:
              gt_values[key] = np.asarray(gt_values[key])
              pred_values[key] = np.asarray(pred_values[key])

          results = {}
          for key in parameters:
              gt = gt_values[key]
              pred = pred_values[key]

              difference = pred - gt                                # Signed error: reconstructed - ground truth 
              if key == "PA":                                       # Position angle is periodic over 180 degrees
                  difference = (difference + 90.0) % 180.0 - 90.0

              absolute_error = np.abs(difference)
              results[key] = {
                  "GT": gt,
                  "Predicted": pred,
                  "signed_error": difference,
                  "absolute_error": absolute_error,

                  # Statistics
                  "mean_error": np.mean(difference),
                  "std_error": np.std(difference, ddof=1),
                  "MAE": np.mean(absolute_error),
                  "std_absolute_error": np.std(absolute_error, ddof=1),
                  "RMSE": np.sqrt(np.mean(difference**2))
              }

          return results
          
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
          
          
