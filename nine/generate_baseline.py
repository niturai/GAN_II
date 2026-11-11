import numpy as np
from aperture import aper
from obstime import observe
from obspoint import vary_base
import matplotlib.pyplot as plt
import cv2

ap = aper()
obs = observe()
vb = vary_base()

# observation time for Intensity Interferometry
start = [2460311.20167]
end = [2460311.70833]
step = obs.obslen(start, end, 0.0104167)
jd = obs.julday(start, end, step)


# Visualization of telescope position, name and baseline's name
latitude = [(28, 18, 03.69), (28, 18, 02.43), (28, 18, 08.52), 
            (28, 18, 08.31), (28, 18, 08.73), (28, 18, 14.92), 
            (28, 18, 15.56), (28, 17, 57.45), (28, 18, 02.75)] # Latitude of each telescope (degree, minutes and second)

lat = [vb.raddeg(*x) for x in latitude]                        # Latitude in radian

longitude = [(16, 30, 28.69), (16, 30, 23.78), (16, 30, 29.82), 
             (16, 30, 23.90), (16, 30, 17.63), (16, 30, 24.88), 
             (16, 30, 18.56), (16, 30, 31.34), (16, 30, 33.98)]# Longitude of each telescope (degree, minutes and second)

lon = [vb.raddeg(*x) for x in longitude]                       # Longitude in radian
             
Tname = ['ASTRI-1', 'ASTRIT-2', 'ASTRIT-3', 'ASTRIT-4', 
        'ASTRI-5', 'ASTRI-6', 'ASTRI-7', 'ASTRI-8', 'ASTRI-9'] # the telescope's name
nbase = ap.basename(Tname)                                     # The baseline name

latitude = np.array([vb.deg(*x) for x in latitude])            # latitude in degree
longitude = np.array([vb.deg(*x) for x in longitude])          # longitude in degree
Tpos = latitude + longitude * 1j                               # Creating a NumPy array of telescopes with complex numbers (latitude & longitude)
ang = np.exp(1.6j)                                             # An angle to rotate the telescope's position
Tpos = [ang*x for x in Tpos]                                   # Exact Telescope's look from top

# the given source position RA and Dec
r = vb.radhr(1, 43, 39)
de = vb.raddeg(50, 41, 19.43)
R = 6.38e6                                                     # radius of Earth in meter
lam = 4e-7                                                     # observational wavelength in meter
vb.ps_para(rac=r, dec=de, R=R, lam=lam)
ut, vt, wt = vb.rotbase(lat, lon, jd)                          # Array of array of the baseline position according to earth rotation

# visualize the telescope and track of baselines
figname = '../result/nine/telescope.png'                                      # Name of file to be saved for telescope's visualization
fname = '../result/nine/track/track'
dark_colors = ap.colors()
color = dark_colors[:len(ut)]                                  # Limit to first N number of dark colors (Number of baseline)
ap.telescope(Tpos, Tname, color, figname)                      # The telescope's position
#tr = ap.track(ut, vt, step, color, fname)

# plot the covered (u,v) plane with baselines (comment out the lines for the visualizations of baseline only)
plt.close()
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = [6,6]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
st = np.cumsum(step)
dark_colors = ap.colors()
color = dark_colors[:len(ut)]                                  # Limit to first N number of dark colors (Number of baseline)
for k in range(0, len(ut), 1):
    #plt.plot(ut[k,:], vt[k,:], '.', color='black')
    #plt.plot(ut[k,:], vt[k,:], 'o', markeredgecolor='blue', markersize=6, markeredgewidth=1.5, color=color[k])
    plt.plot(ut[k,:], vt[k,:], 'o', markersize=4, color=color[k])
    
#plt.axis('off')
plt.gca().set_aspect('equal')
plt.xlabel('Along East in meter')
plt.ylabel('Along North in meter')
plt.title('Covered Observational (u, v) Plane', fontweight='bold')
plt.savefig('../result/nine/base/baseline_1.png')
#plt.savefig('../result/nine/base/baseline.png')#, bbox_inches='tight', pad_inches=0, dpi=500)


# Convert the observational plane to grayscale and re-arrange pixels and save the baseline in .npy format
base = plt.imread("../result/nine/base/baseline.png")
image_size = 128  # px
mask = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
mask = np.where(mask < 1, 1, 0).astype(np.float32)                                    # Return elements chosen from x or y depending on condition (condition, x, y)
mask = cv2.resize(mask, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA) # gives the best results
mask = np.where(mask > 0, 1, 0)
np.save("../result/nine/base_npy/base", mask)                                                        # Save mask as numpy array

print("finished")
