# -*- coding: utf-8 -*-
"""
Created on Jun 2022

@author: Mahsa Farahani
mahsafarahani971@gmail.com

KNN classifier for remote sensing images

"""

# import libraries
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt
import time

# to see the run time
start_time = time.time()

# read satellite image data and the target data
with rio.open("input your satellite image address here") as img:
    data_image = img.read()

# read train image
with rio.open("input your train image address here") as tar:
    data_target = tar.read()

# shape and size    
d, r, c = data_image.shape
n = r*c

# n*d data
data = np.zeros((n,d))
for i in range(d):
    data[:,i] = data_image[i,:,:].flatten()
    
target = data_target.flatten()

del data_image, data_target

# normalizing data between [0 1] (optional)
for i in range(d):
    minn = np.min(data[:,i])
    maxx = np.max(data[:,i])
    data[:,i] = (data[:,i] - minn) / (maxx - minn)

# train image is an image of dim r*c, some pixels are labled and the rest are zero
x_train = data[target!=0,:]
y_train = target[target!=0]

# negibors range (to find best k), you can insert any k numbers you want
nk_min = 2
nk_max = 7 #exclusive
k_list = list(range(nk_min,nk_max)) # a group list of desired neighbor numbers
n_k = len(k_list)
classified = np.zeros((n,n_k))

# find distance of every data to train data and find the closests
for x in range(n):
    xsub = data[x,:] - x_train
    dist = np.sqrt(np.sum(xsub ** 2, 1))
    sortedlabels = y_train[np.argsort(dist)]
    for nk in range(n_k): # finding best k
        knlabels = sortedlabels[range(0,k_list[nk])]
        klabs, kcounts = np.unique(knlabels, return_counts=True)
        classified[x, nk] = klabs[np.argmax(kcounts)]
        
# Plot and show one one of classified images
plt.figure()
plt.imshow(classified[:,0].reshape((r,c)))

print("--- %s seconds ---" % (time.time() - start_time))

# open and read test image for validation
with rio.open("input your test image address here") as tes:
    data_test = tes.read()
    
y_te = data_test.flatten()

labels = np.unique(target)[1:] # to remove no label data i.e. 0
n_cl = labels.size

# Validation:
class_acc = np.zeros((n_cl,n_k))
n_classified = np.zeros((n_cl,n_k))
n_validate = np.zeros((n_cl,1))
overal_acc = np.zeros((n_k,1))
for j in range(n_k):
    for i in range(n_cl):
        n_classified[i,j] = len(np.where((classified[:,j]==labels[i]) & (
            y_te==labels[i]))[0])
        n_validate[i,0] = len(np.where( y_te==labels[i])[0])
        class_acc[i,j] = n_classified[i,j]/n_validate[i,0] * 100
    
    overal_acc[j,0] = sum(n_classified[:,j])/sum(n_validate) * 100
        
    print("for k = {}, overal accuracy is {}".format(k_list[j],overal_acc[j,0]))

# best k
bk = k_list[np.argmax(overal_acc)]
print("Best k for this data is: ", bk)

# Plot the classified image
plt.figure()
plt.imshow(classified[:,bk-1].reshape((r,c)))
