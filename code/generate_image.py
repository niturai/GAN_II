import matplotlib.pyplot as plt
from ellipse import ellipsoid
import numpy as np
import os


ep = ellipsoid()      

# Create folders if they don't exist
os.makedirs('ellip_image', exist_ok=True)
os.makedirs('ellip_npy', exist_ok=True) 
 
# fixed the parameters
sx, sy, x, y = ep.grids(1e-10, 512, 1e-6)   # N = 512, shape of the image (255, 255) as np.arange is used in grids
rad = np.arange(3e-9, 1.6e-8, 2e-10)
inc = np.arange(0, 2*np.pi, np.pi/4)
pa = np.arange(0, 2*np.pi, np.pi/4)
sq = np.arange(0.5, 1, 0.1)
pic = 0
for r in rad:
    for i in inc:
        # Add a small deviation, to dodge the singularities
        if (i == (1 / 2) * np.pi) or (i == (3 / 2) * np.pi):
           i += 0.07
        if (i == 0) or (i == np.pi):
           ellipse = ep.ellip(sx, sy, r, i, 0, 1)
           plt.imsave('ellip_image/ellipse' + str(pic) + '.jpg', ellipse)
           np.save('ellip_npy/ellipse' + str(pic) + '.npy', ellipse)
           pic += 1
        else:
           for p in pa:
               for s in sq:
                   ellipse = ep.ellip(sx, sy, r, i, p, s)
                   plt.imsave('ellip_image/ellipse' + str(pic) + '.jpg', ellipse)
                   np.save('ellip_npy/ellipse' + str(pic) + '.npy', ellipse)
                   pic += 1
                   
print("finished")
