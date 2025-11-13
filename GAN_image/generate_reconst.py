import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import glob
import os 
import re


# Load all .npy file
main = "testing_image/testing_image/"

data_path = os.path.join(main, "image_*_input.npy")
files1 = glob.glob(data_path)
files1.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])  # Natural sort by numeric part

data_path = os.path.join(main, "image_*_target.npy")
files2 = glob.glob(data_path)
files2.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])  # Natural sort by numeric part

data_path = os.path.join(main, "image_*_prediction.npy")
files3 = glob.glob(data_path)
files3.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])  # Natural sort by numeric part

data_path = os.path.join(main, "image_*_difference.npy")
files4 = glob.glob(data_path)
files4.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])  # Natural sort by numeric part

Inp = []
tar = []
pre = []
diff = []
# Loop over both lists together
for f1, f2, f3, f4 in zip(files1, files2, files3, files4):
    data1 = np.load(f1)   # load Input data
    data2 = np.load(f2)   # load target Image
    data3 = np.load(f3)   # load prediction Image
    data4 = np.load(f4)   # load difference
    Inp.append(data1)
    tar.append(data2)
    pre.append(data3)
    diff.append(data4)

# Convert lists to arrays
Inp = np.array(Inp)
tar = np.array(tar)
pre = np.array(pre)
diff = np.array(diff)

plt.rcParams.update({'font.size': 8})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.figure(figsize=(13, 3)) 
titles = ['Observed Data', 'Ground Truth Image', 'Predicted Image', 'Difference']
N_images = 4
counter = 0

for i in range(len(Inp)):
    display_list = [Inp[i], tar[i], pre[i]]
                                        
    # --- Plot input, target, predicted as images ---
    for j in range(3):
        plt.subplot(1, 4, j+1) 
        plt.title(titles[j], fontweight='bold') 
        plt.imshow(display_list[j]*0.5 + 0.5) 
        plt.colorbar(shrink=0.9) 
        plt.gca().set_aspect('equal') 
        plt.axis('off') 
        plt.tight_layout()            
    
    plt.subplots_adjust(wspace=0.001)
    
    # --- Plot difference as 1D residual profile ---
    residuals = np.mean(diff[i], axis=0)   # collapse rows -> 1D profile
    plt.subplot(1, 4, 4)
    plt.title("Residuals of Images", fontweight='bold')
    plt.plot(residuals, color='red')
    plt.axhline(0, color='black', linestyle='--')
    #plt.xlabel("Pixel index")
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    #plt.ylabel("Mean Residuals")
    plt.tight_layout()
    plt.savefig(f"testing_image/testing_image/image_{counter}.png", dpi=300)
    counter += 1
    plt.clf()
    




