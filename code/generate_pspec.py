import matplotlib.pyplot as plt
from image_noise import Pimage
from scipy import fft
import numpy as np
import random
import cv2
import os


ps = Pimage()
image_size = 128                                                                                      # px
N_tele = 4                                                                                            # number of telescope
alpha = 0.005                                                                                         # Salt and Pepper Noise probability
save_images = True

"""
Script that calculates the Power spectrum (2D-FFT) of an image (i.e. a stellar source). Then automatically creates training, testing and validation datasets. Includes addition of Salt and Pepper noise. Output: test, train and validation folders contaning merged images (stellar objects in sky and on observational plane), ready to be used in the model.
"""

# Create the folder if it doesn't exist
os.makedirs("val", exist_ok=True)
os.makedirs("test", exist_ok=True)
os.makedirs("train", exist_ok=True)

base = np.load("base_npy/base.npy")                                                                   # the baseline in .npy format
path_in = "ellip_npy/"                                                                                # the ellipsoid image in .npy format
counter = 0
for filename in os.listdir(path_in):
    f = os.path.join(path_in, filename)                                                               # load image
    image_original = np.load(f)                                                                       # input image
    img_org = image_original.copy()                                                                   # for ground truth
    ground_truth = cv2.resize(img_org, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)  # Resize the image_size of ground truth (stellar object)
    
    image_original = ps.sap_noise(img_org, alpha)                                                     # ground truth with salt and pepper noise
    img_ed = cv2.resize(image_original, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA) # Resize the image_size and subtract the mean for noisy image
    img_ed = img_ed - np.mean(img_ed)                                                                 # Subtracting the mean from each pixel in the image centers the data around zero (helps center and 
                                                                                                      # normalize the data, remove bias, improve network convergence, and reduce covariate shift).   
    img_fft = fft.fftshift(fft.fft2(fft.fftshift(img_ed)))                                            # calculate 2D FFT for noisy image (power spectrum)
    fft_argument = np.abs(img_fft)                                                                    # the absolute value of fourier transform
    
    fft_argument = np.multiply(fft_argument, base)                                                    # sparse sampling (the power spectrum for given baselines)
    img_fft_norm = fft_argument/np.max(fft_argument)                                                  # normalization of the power spectrum for given baselines
    
    # the images are on float ([0,1]), but cv2 requires integer ([0, 255]):
    img_fft_norm = cv2.convertScaleAbs(img_fft_norm, alpha=(255.0))
    #img_fft_norm = img_fft_norm/img_fft_norm.max()
    ground_truth = cv2.convertScaleAbs(ground_truth, alpha=(255.0))
    #ground_truth = ground_truth/ground_truth.max()
    
    combined_image = ps.add_image(ground_truth, img_fft_norm)                                         # combine ground truth and input image (power spectrum)
    
    image_name = filename[:-4]                                                                        # save or display images > use cv2 instead of matplotlib, this always saves as (64,64,4)
    if save_images:
       # Choose randomly where to save
       rnd = random.random()  # returns float in [0,1)
       plt.figure(figsize=(8, 8))
       if rnd < 0.1:  # 10% for validation
            save_path = os.path.join("val", image_name + ".jpg")
            '''
            im = plt.imshow(combined_image, cmap='gray', aspect='equal')
            cbar = plt.colorbar(im, shrink=0.45, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=12, width=1.5)
            plt.axis('off')
            plt.tight_layout(pad=1.0)
            plt.savefig(f'val/{image_name}.jpg', dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.clf()
            plt.close()
            '''
       elif rnd < 0.2:  # next 10% for testing
            save_path = os.path.join("test", image_name + ".jpg")
            '''
            im = plt.imshow(combined_image, cmap='gray', aspect='equal')
            cbar = plt.colorbar(im, shrink=0.45, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=12, width=1.5)
            plt.axis('off')
            plt.tight_layout(pad=1.0)
            plt.savefig(f'test/{image_name}.jpg', dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.clf()
            plt.close()
            '''
       else:  # remaining 80% for training
            '''
            im = plt.imshow(combined_image, cmap='gray', aspect='equal')
            cbar = plt.colorbar(im, shrink=0.45, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=12, width=1.5)
            plt.axis('off')
            plt.tight_layout(pad=1.0)
            plt.savefig(f'train/{image_name}.jpg', dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.clf()
            plt.close()
            '''
            save_path = os.path.join("train", image_name + ".jpg")

       cv2.imwrite(save_path, combined_image)

    else:
            plt.imshow(ground_truth)
            print("Original image: ", np.mean(ground_truth))
            plt.show()
            plt.imshow(img_fft_norm)
            print("FF2D image: ", np.mean(img_fft_norm))
            #plt.show()

            if counter == 2:
                break

    counter += 1

    if save_images:
        print(f"{counter} images successfully converted.")
    else:
        print(f"{counter} images not saved")
    
    counter += 1
if save_images:
    print(f"{counter} images successfully converted.")
else:
    print(f"{counter} images not saved")
   
print("finished")   
