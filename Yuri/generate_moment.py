import matplotlib.pyplot as plt
from moment import mom
import numpy as np
import glob
import os 
import re

plt.rcParams.update({'font.size': 10})
plt.rcParams["figure.figsize"] = [12,6]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# The Directory of all images  
img_dir = "testing_image/testing_image_npy/" 
data_path = os.path.join(img_dir,'*.npy') 
files = glob.glob(data_path)
files.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)',x)])

ground_img = []
pred_img = [] 
for f1 in files:
    data = np.load(f1)
    ground_img.append(data[1])
    pred_img.append(data[2])

gr_img = np.array(ground_img)       # The ground truth image of stellar object
pr_img = np.array(pred_img)         # The predicted image of stellar object with trained GAN

# for visualisation of ground truth and predicted image
pic = 0
for i in range(len(pr_img)):
    plt.subplot(1,2,1)
    plt.imshow(gr_img[i])
    plt.subplot(1,2,2)
    plt.imshow(pr_img[i])
    plt.savefig('result/image/image' + str(pic) + '.png')
    #plt.show()
    pic += 1
   
# Calculate moment only for one image
im_g = gr_img[38]
im_p = pr_img[38]

md = mom()

# the moments
M00_g, mx_g, my_g, mu11_g, mu20_g, mu02_g, mu30_g, mu03_g, mu21_g, mu12_g = md.mom_def(im_g)
M00_p, mx_p, my_p, mu11_p, mu20_p, mu02_p, mu30_p, mu03_p, mu21_p, mu12_p = md.mom_def(im_p)

# print the monopole and central moments
print('Monopole the total intensity for ground truth and predicted image: ', (M00_g, M00_p))
print('Intensity Centroid for ground truth and predicted image:', (mx_g, my_g), (mx_p, my_p))
print('second order central moment for ground truth and predicted image', (mu11_g, mu20_g, mu02_g), (mu11_p, mu20_p, mu02_p))
print('third order central moment for ground truth and predicted image', (mu30_g, mu03_g, mu21_g, mu12_g), (mu30_p, mu03_p, mu21_p, mu12_p))

# Information about the size and shape of the image using third-order central moment
print('the ellipse parameters for the ground truth and predicted image with GAN')
alpha_g, a_g, b_g, e_g, A_g = md.akar(mu11_g, mu20_g, mu02_g)
alpha_p, a_p, b_p, e_p, A_p = md.akar(mu11_p, mu20_p, mu02_p)

angle_g = np.rad2deg(alpha_g) # Convert the orientation into degree
angle_p = np.rad2deg(alpha_p)
print("the orientation of the ground ellipse in degree = {}°".format(angle_g))
print("the orientation of the generated ellipse in degree = {}°".format(angle_p))


# Information about the surface of the image using third-order central moment
Sx_g = mu30_g/mu20_g**(3/2) # skweness along x direction
Sy_g = mu03_g/mu02_g**(3/2) # skweness along y direction
Sx_p = mu30_p/mu20_p**(3/2)
Sy_p = mu03_p/mu02_p**(3/2)

print('Skewness of the ground ellipse along x and y direction :', (Sx_g, Sy_g))
print('Skewnessthe of the generated ellipse along x and y direction :', (Sx_p, Sy_p))
print("Third order central moments of the ground and generated ellipse (mu21, mu12):", (mu21_g, mu12_g), (mu21_p, mu12_p))

# visualize the ground truth and predicted image
plt.subplot(1,2,1)
plt.imshow(im_g)
plt.title('The ellipse as ground truth', fontweight='bold')
plt.subplot(1,2,2)
plt.imshow(im_p)
plt.title('The reconstructed ellipse with GAN', fontweight='bold')
plt.savefig('result/images.png')
#plt.show()
plt.close()


# Generate ellipse points
x_g, y_g = md.prakar(mx_g, my_g, alpha_g, a_g, b_g)
x_p, y_p = md.prakar(mx_p, my_p, alpha_p, a_p, b_p)


# visualize the ground truth and predicted image with moments and reconstructed size using it
plt.subplot(1,2,1)
plt.imshow(im_g)
plt.plot(x_g, y_g, label='Ellipse', color='red', linewidth = 3)
plt.plot(mx_g, my_g, 'o', color='magenta',markersize=10)
plt.arrow(mx_g, my_g, Sx_g, 0, color='darkcyan', head_width=2, head_length=15)
plt.arrow(mx_g, my_g, 0, Sy_g, color='blue', head_width=2, head_length=15)
plt.title('The ellipse as ground truth', fontweight='bold')
plt.subplot(1,2,2)
plt.imshow(im_p)
plt.plot(x_p, y_p, label='Ellipse', color='red', linewidth = 3)
plt.plot(mx_p, my_p, 'o', color='magenta',markersize=10)
plt.arrow(mx_p, my_p, Sx_p, 0, color='darkcyan', head_width=2, head_length=15)
plt.arrow(mx_p, my_p, 0, Sy_p, color='blue', head_width=2, head_length=15)
plt.title('The reconstructed ellipse with GAN', fontweight='bold')
plt.savefig('result/reconstruction.png')
#plt.show()
plt.close()

plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = [8,8]
plt.rcParams['axes.facecolor']='ivory'
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
# visualise the moments of each generated images
list_g = []
list_p = []
for i in range(len(gr_img)):
    im_g = gr_img[i]
    im_p = pr_img[i]
    list_g.append(md.mom_def(im_g))
    list_p.append(md.mom_def(im_p))

Mth_g = []
Mth_p = []
m_name = ['Monopole', 'X-Centroid', 'Y-Centroid', '2nd Moment $\mu_{11}$', '2nd Moment $\mu_{20}$', '2nd Moment $\mu_{02}$', '3rd Moment $\mu_{30}$','3rd Moment $\mu_{03}$','3rd Moment $\mu_{21}$', '3rd Moment $\mu_{12}$']
pic = 0
for i in range(len(list_g[0])):
    Mth_g = [M[i] for M in list_g]
    Mth_p = [M[i] for M in list_p]
    for j in range(len(Mth_g)):
        plt.plot(Mth_g[j], Mth_p[j], 'o', markeredgecolor='blue', markersize=8, markeredgewidth=1.5)

    plt.xlabel('The Ground Truth')
    plt.ylabel('The Predicted Image')
    plt.title(f'{m_name[i]} for Stellar object', fontweight='bold')
    plt.gca().set_aspect('equal')
    plt.savefig('result/moments/mom' + str(pic) + '.png')
    #plt.show()
    plt.clf()
    pic += 1
    

# the reconstructed image with moments
w = md.ellip(im_p, mx_p, my_p, alpha_p, a_p, e_p)
plt.imshow(w)
plt.title(f'The reconstructed Stellar object using moments of 2nd order', fontweight='bold')
plt.savefig('result/ellipse.png')
#plt.show()
plt.clf()
