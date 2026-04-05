import matplotlib.pyplot as plt
from moment import mom
import numpy as np
import glob
import os 
import re

md = mom()

# The Directory of all images  
gr_img6 = "result/six/testing_image/"
gr_img9 = "result/nine/testing_image/testing_image/"
data_path6 = os.path.join(gr_img6, "image_*_target.npy")
data_path9 = os.path.join(gr_img9, "image_*_target.npy")
files16 = glob.glob(data_path6)
files16.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])  # Natural sort by numeric part
files19 = glob.glob(data_path9)
files19.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])  # Natural sort by numeric part

data_path6 = os.path.join(gr_img6, "image_*_prediction.npy")
data_path9 = os.path.join(gr_img9, "image_*_prediction.npy")
files26 = glob.glob(data_path6)
files26.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])  # Natural sort by numeric part
files29 = glob.glob(data_path9)
files29.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])  # Natural sort by numeric part

ground_img6 = []
pred_img6 = []
ground_img9 = []
pred_img9 = []
# Loop over both lists together
for f16, f26, f19, f29 in zip(files16, files26, files19, files29):
    data16 = np.load(f16)   # load ground truth
    data26 = np.load(f26)   # load prediction
    
    data19 = np.load(f19)   # load ground truth
    data29 = np.load(f29)   # load prediction
    
    ground_img6.append(data16.squeeze())
    pred_img6.append(data26.squeeze())
    
    ground_img9.append(data19.squeeze())
    pred_img9.append(data29.squeeze())

# Convert lists to arrays
gr_img6 = np.array(ground_img6)
pr_img6 = np.array(pred_img6)

gr_img9 = np.array(ground_img9)
pr_img9 = np.array(pred_img9)

print(gr_img6.shape)

# Calculate the Scatter of centroid and moments for ground truth and predicted image
SixSc, SixS11, SixS20, SixS02, SixS12, SixS21, SixS30, SixS03 = md.mom_cal(gr_img6, pr_img6)
nineSc, nineS11, nineS20, nineS02, nineS12, nineS21, nineS30, nineS03 = md.mom_cal(gr_img9, pr_img9)

print('6Sc, 6S11, 6S20, 6S02, 6S12, 6S21, 6S30, 6S03', SixSc, SixS11, SixS20, SixS02, SixS12, SixS21, SixS30, SixS03)
print('9Sc, 9S11, 9S20, 9S02, 9S12, 9S21, 9S30, 9S03', nineSc, nineS11, nineS20, nineS02, nineS12, nineS21, nineS30, nineS03)

# visualise the moments of each generated images
list_g6 = []
list_p6 = []
list_g9 = []
list_p9 = []
for i in range(len(gr_img6)):
    gr6 = gr_img6[i]
    pr6 = pr_img6[i]
    gr9 = gr_img9[i]
    pr9 = pr_img9[i]
    
    im_g6 = (gr6*0.5 + 0.5)             # (128,128) (the scaling factor is for 0 to 1 from -1 to 1)
    im_p6 = (pr6*0.5 + 0.5)             # (128,128)
    
    im_g9 = (gr9*0.5 + 0.5)             # (128,128) (the scaling factor is for 0 to 1 from -1 to 1)
    im_p9 = (pr9*0.5 + 0.5)             # (128,128)
       
    list_g6.append(md.mom_def(im_g6))
    list_p6.append(md.mom_def(im_p6))
    
    list_g9.append(md.mom_def(im_g9))
    list_p9.append(md.mom_def(im_p9))

Mth_g6 = []
Mth_p6 = []
Mth_g9 = []
Mth_p9 = []
m_name = ['Monopole', 'X-Centroid', 'Y-Centroid', '2nd Moment $\mu_{11}$', 
          '2nd Moment $\mu_{20}$', '2nd Moment $\mu_{02}$', '3rd Moment $\mu_{30}$', 
          '3rd Moment $\mu_{03}$','3rd Moment $\mu_{21}$', '3rd Moment $\mu_{12}$']
pic = 0
for i in range(len(list_g6[0])):
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["figure.figsize"] = [6,6]
    plt.rcParams['axes.facecolor']='ivory'
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    Mth_g6 = np.array([M[i] for M in list_g6])
    Mth_p6 = np.array([M[i] for M in list_p6])
    Mth_g9 = np.array([M[i] for M in list_g9])
    Mth_p9 = np.array([M[i] for M in list_p9])

    plt.plot(Mth_g6, Mth_p6, '.', color='green', markersize=9, label='Six Telescope')
    plt.plot(Mth_g9, Mth_p9, '.', color='red', markersize=9, label='Nine Telescope')

    plt.legend(loc='upper left')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted Image')
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.title(f'{m_name[i]} for Stellar object', fontweight='bold')
    plt.tight_layout()
    plt.savefig('result/moments/mom' + str(pic) + '.png')
    plt.clf()
    pic += 1
    plt.close()
    

