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

# telescope position, name, baseline name and baseline length
# the position of Veritas Observatory
la = vb.raddeg(31, 40, 30) 
lo = vb.raddeg(110, 57, 7)

# coordinates for telescope
x = np.linspace(-60, 150, 500)
y = np.linspace(-60, 85, 500)
Tname = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6'] 
Tpos = [135 - 20j, 40 - 50j, 0 + 60j, -40 + 5j, 50 + 20j, 90+65j]
ang = np.exp(0.1j)
Tpos = [ang*x for x in Tpos]
fname = '../result/six/telescope.png'
ap.telescope(Tpos, Tname, fname, x, y, 6)

# baseline name and length (x, y)
tel, base = ap.baseline(Tpos, Tname) 
x = np.real(base) 
y = np.imag(base)
z = 1e-6

# a single star at the position of source Spica
r = vb.radhr(13, 25, 11.579)
de = vb.raddeg(-11, 9, 40.75)

# variational baseline according to the earth rotation
vb.ps_para(lat=la, lon=lo, rac=r, dec=de)
dist = []                    # the baseline position according to earth rotation
for i in range(len(x)):
    dist.append(vb.rotbase(x[i], y[i], z, jd))

distance = np.array(dist)
xt = distance[:,0]           # baselines in east direction
yt = distance[:,1]           # baselines in north direction
zt = distance[:,2]           # baseline in up

# plot the covered (u,v) plane with baselines (comment out the lines for the visualizations of baseline only)
plt.close()
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = [6,6]
plt.rcParams['axes.facecolor']='ivory'
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
st = np.cumsum(step)
dark_colors = ap.colors()
color = dark_colors[:len(xt)]                                  # Limit to first N number of dark colors (Number of baseline)
for k in range(0, len(xt), 1):
    #plt.plot(xt[k,:], yt[k,:], '.', color='black')
    plt.plot(xt[k,:], yt[k,:], 'o', markersize=4, color=color[k])
    
#plt.axis('off')
#plt.gca().set_aspect('equal')
plt.xlabel('Along East in meter')
plt.ylabel('Along North in meter')
plt.title('Covered Observational (u, v) Plane', fontweight='bold')
plt.savefig('../result/six/base/baseline_1.png')
#plt.savefig('../result/six/base/baseline.png')#, bbox_inches='tight', pad_inches=0, dpi=500)


# Convert the observational plane to grayscale and re-arrange pixels and save the baseline in .npy format
base = plt.imread("../result/six/base/baseline.png")
image_size = 128  # px
mask = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
mask = np.where(mask < 1, 1, 0).astype(np.float32)                                    # Return elements chosen from x or y depending on condition (condition, x, y)
mask = cv2.resize(mask, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA) # gives the best results
mask = np.where(mask > 0, 1, 0)
np.save("../result/six/base_npy/base", mask)                                                        # Save mask as numpy array

print("finished")
