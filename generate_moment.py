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
SixSc, SixS11, SixS20, SixS02, SixS12, SixS21, SixS30, SixS03, error = md.mom_cal(gr_img6, pr_img6)
nineSc, nineS11, nineS20, nineS02, nineS12, nineS21, nineS30, nineS03, error = md.mom_cal(gr_img9, pr_img9)

print('6Sc, 6S11, 6S20, 6S02, 6S12, 6S21, 6S30, 6S03', SixSc, SixS11, SixS20, SixS02, SixS12, SixS21, SixS30, SixS03)
print('9Sc, 9S11, 9S20, 9S02, 9S12, 9S21, 9S30, 9S03', nineSc, nineS11, nineS20, nineS02, nineS12, nineS21, nineS30, nineS03)
print("Centroid error:")
print("Mean =", errors["centroid"]["mean_error"])
print("SD   =", errors["centroid"]["std_error"])
print("RMSE =", errors["centroid"]["rmse"])

print("\nSecond-order moment mu11:")
print("Mean signed error =", errors["mu11"]["mean_error"])
print("SD                =", errors["mu11"]["std_error"])
print("MAE               =", errors["mu11"]["mae"])
print("RMSE              =", errors["mu11"]["rmse"])

print("\nSecond-order moment mu20:")
print("Mean signed error =", errors["mu20"]["mean_error"])
print("SD                =", errors["mu20"]["std_error"])
print("MAE               =", errors["mu20"]["mae"])
print("RMSE              =", errors["mu20"]["rmse"])

print("\nSecond-order moment mu02:")
print("Mean signed error =", errors["mu02"]["mean_error"])
print("SD                =", errors["mu02"]["std_error"])
print("MAE               =", errors["mu02"]["mae"])
print("RMSE              =", errors["mu02"]["rmse"])

print("\nthird-order moment mu21:")
print("Mean signed error =", errors["mu21"]["mean_error"])
print("SD                =", errors["mu21"]["std_error"])
print("MAE               =", errors["mu21"]["rmse"])
print("RMSE              =", errors["mu21"]["mae"])

print("\nthird-order moment mu12:")
print("Mean signed error =", errors["mu12"]["mean_error"])
print("SD                =", errors["mu12"]["std_error"])
print("MAE               =", errors["mu12"]["rmse"])
print("RMSE              =", errors["mu12"]["mae"])

print("\nthird-order moment mu30:")
print("Mean signed error =", errors["mu30"]["mean_error"])
print("SD                =", errors["mu30"]["std_error"])
print("MAE               =", errors["mu30"]["rmse"])
print("RMSE              =", errors["mu30"]["mae"])

print("\nthird-order moment mu03:")
print("Mean signed error =", errors["mu03"]["mean_error"])
print("SD                =", errors["mu03"]["std_error"])
print("MAE               =", errors["mu03"]["rmse"])
print("RMSE              =", errors["mu03"]["mae"])

errors_N6 = md.parameter_errors(gr_img6, pr_img6)                  # Calculate errors for N6
errors_N9 = md.parameter_errors(gr_img9, pr_img9)                  # Calculate errors for N9

print("\nParameter reconstruction errors")
print("-" * 75)
print("{:<12} {:<25} {:<25}".format("Parameter", "N6 (MAE ± Std.)", "N9 (MAE ± Std.)"))
print("-" * 75)

for key in ["PA", "a", "b", "e"]:
    n6_mae = errors_N6[key]["MAE"]
    n6_std = errors_N6[key]["std_absolute_error"]

    n9_mae = errors_N9[key]["MAE"]
    n9_std = errors_N9[key]["std_absolute_error"]

    print("{:<12} {:<25} {:<25}".format(key, f"{n6_mae:.4f} ± {n6_std:.4f}", f"{n9_mae:.4f} ± {n9_std:.4f}"))

print("-" * 75)

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
    

