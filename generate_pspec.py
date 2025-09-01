import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from image_noise import Pimage
import cv2
import os


ps = Pimage()
image_size = 128                                                                                      # px
N_tele = 4                                                                                            # number of telescope
alpha = 0.005                                                                                         # Salt and Pepper Noise probability
save_images = True

"""
Script that calculates the Power spectrum (2D-FFT) of an image (i.e. a stellar source). Then automatically creates training, testing and validation datasets. Includes addition of Salt and Pepper noise. Output: test, train and validation folders contaning merged images (stellar objects in sky and on observational plane), ready to be used in the pix2pix model.
"""

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
            if counter < 150:
               '''
               im = plt.imshow(combined_image, cmap='gray')
               plt.colorbar(shrink=0.55)
               plt.gca().set_aspect('equal')
               plt.axis('off')
               plt.savefig('val/' + image_name + '.jpg')
               plt.clf()
               plt.close()
               '''
               cv2.imwrite('val/' + image_name + '.jpg', combined_image)
            elif (counter >= 150) and (counter < 300):
               '''
               im = plt.imshow(combined_image, cmap='gray')
               plt.colorbar(shrink=0.55)
               plt.gca().set_aspect('equal')
               plt.axis('off')
               #plt.show()
               plt.savefig('test/' + image_name + '.jpg')
               plt.clf()
               plt.close()
               '''
               cv2.imwrite('test/' + image_name + '.jpg', combined_image)
            else:
               '''
               im = plt.imshow(combined_image, cmap='gray')
               plt.colorbar(shrink=0.55)
               plt.gca().set_aspect('equal')
               plt.axis('off')
               plt.savefig('train/' + image_name + '.jpg')
               plt.clf()
               plt.close()
               '''
               cv2.imwrite('train/' + image_name + '.jpg', combined_image)

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
   
    
