import matplotlib.pyplot as plt
from moment import mom
import numpy as np
import glob
import os 
import re

md = mom()

# The Directory of all images  
gr_img = "testing_image/testing_image/"
data_path = os.path.join(gr_img, "image_*_target.npy")
files1 = glob.glob(data_path)
files1.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])  # Natural sort by numeric part

pr_img = "testing_image/testing_image/"
data_path = os.path.join(pr_img, "image_*_prediction.npy")
files2 = glob.glob(data_path)
files2.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])  # Natural sort by numeric part

ground_img = []
pred_img = []
# Loop over both lists together
for f1, f2 in zip(files1, files2):
    data1 = np.load(f1)   # load ground truth
    data2 = np.load(f2)   # load prediction
    ground_img.append(data1.squeeze())
    pred_img.append(data2.squeeze())

# Convert lists to arrays
gr_img = np.array(ground_img)
pr_img = np.array(pred_img)

plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = [6,6]
plt.rcParams['axes.facecolor']='ivory'
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# visualise the moments of each generated images
list_g = []
list_p = []
for i in range(len(gr_img)):
    gr = gr_img[i]
    pr = pr_img[i]
    im_g = (gr*0.5 + 0.5)   # (128,128) (the scaling factor is for 0 to 1 from -1 to 1)
    im_p = (pr*0.5 + 0.5)   # (128,128)
    list_g.append(md.asym(im_g))
    list_p.append(md.asym(im_p))


for j in range(len(list_g)):
    plt.plot(list_g[j], list_p[j], 'o', markeredgecolor='blue', markersize=8, markeredgewidth=1.5)

plt.xlabel('Ground Truth')
plt.ylabel('Predicted Image')
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.title(f'Asymmetry for Stellar object', fontweight='bold')
#plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig('result/moments/asymmetry.png')
#plt.show()
    
