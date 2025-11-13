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
    ground_img.append(data1)
    pred_img.append(data2)

# Convert lists to arrays
gr_img = np.array(ground_img)
pr_img = np.array(pred_img)
print(gr_img.shape)

plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = [6,6]
plt.rcParams['axes.facecolor']='ivory'
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# visualise the moments of each generated images
list_g = []
list_p = []
for i in range(len(gr_img)):
    im_g = gr_img[i].squeeze()   # (128,128)
    im_p = pr_img[i].squeeze()   # (128,128)
    list_g.append(md.mom_def(im_g))
    list_p.append(md.mom_def(im_p))


os.makedirs("moments", exist_ok=True)

Mth_g = []
Mth_p = []
m_name = ['Monopole', 'X-Centroid', 'Y-Centroid', '2nd Moment $\mu_{11}$', 
          '2nd Moment $\mu_{20}$', '2nd Moment $\mu_{02}$', '3rd Moment $\mu_{30}$', 
          '3rd Moment $\mu_{03}$','3rd Moment $\mu_{21}$', '3rd Moment $\mu_{12}$']
pic = 0
for i in range(len(list_g[0])):
    Mth_g = [M[i] for M in list_g]
    Mth_p = [M[i] for M in list_p]
    for j in range(len(Mth_g)):
        plt.plot(Mth_g[j], Mth_p[j], 'o', markeredgecolor='blue', markersize=8, markeredgewidth=1.5)

    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted Image')
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.title(f'{m_name[i]} for Stellar object', fontweight='bold')
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig('moments/mom' + str(pic) + '.png')
    #plt.show()
    plt.clf()
    pic += 1 
