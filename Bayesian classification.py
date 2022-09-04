
# -*- coding: utf-8 -*-
"""
Created on Jun 2022

@author: Mahsa Farahani
mahsafarahani971@gmail.com

Bayesian classification for remote sensing images

"""

# Import necessary libraries:
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt

# Open data and target images:
with rio.open('input your satellite image address here') as img:
    image_data = img.read()
    
with rio.open('input your target image address here') as trg:
    target_data = trg.read().squeeze()

# Class labels and number of classes:
labels = np.unique(target_data)[1:] # index 1 because label 0 is for those pixels with no information
n_cl = labels.size

# Reshaping image arrays to n*d vectors:
d,r,c = image_data.shape
n = r*c
data = np.zeros((n, d))
for i in range(d):
    data[:,i] = image_data[i,:,:].flatten()
    
target = target_data.flatten()

del image_data, target_data

n_bands = d

# Choosing bands (you can use all bands of rs images but it is better
# to identify best bands by dimension reduction methods like PCA):
# bands = [157,78,49,81,20,27,189,192,116,12,47,71,165,4,9,34,140,147,130,91]
# bands.sort()
# n_bands = len(bands)
# data = data[:,bands]

# split target data to train and test (like "train_test_split" in sklearn)
def splitting_data(x, y, per=0.3):
    # x : data(n*d) 
    # y : target data(n*1) #
    # per: persent for test data (default 30 %)
    n_cl = np.unique(y)[1:].size
    train_list = []
    test_list = []
    for i in range(n_cl):
        q_classes = np.where(target==labels[i])[0]
        ncl = len(q_classes)
        nte = round(per * ncl)
        r_list = list(np.random.choice(range(ncl), nte, replace=False))
        test_list.extend(list(q_classes[r_list]))
        train_list.extend(list(np.delete(q_classes, r_list)))
    
    x_tr = np.zeros((x.shape))
    y_tr = np.zeros((y.shape))
    x_te = np.zeros((x.shape))
    y_te = np.zeros((y.shape))
    x_tr[train_list, :] = x[train_list, :]
    y_tr[train_list] = y[train_list]
    x_te[test_list, :] = x[test_list, :]
    y_te[test_list] = y[test_list]
    
    return x_tr, x_te, y_tr, y_te

x_tr, x_te, y_tr, y_te = splitting_data(data,target)

# plot test and train labal data
plt.figure(1)
plt.imshow(y_tr.reshape((r,c)))
plt.figure(2)  
plt.imshow(y_te.reshape((r,c)))

# prior probabilities
Pc0 = np.zeros((1,n_cl))
for i in range(n_cl):
    Pc0[0,i] = len(np.where( target==labels[i])[0])/n

# Calculating mean and covariance of classes in each band
training_mean = np.zeros((n_cl,n_bands))
training_cov = np.zeros((n_cl,n_bands,n_bands))

for i in range(n_cl):
    tr_data = x_tr[y_tr==labels[i],:]
    training_mean[i,:] = np.mean(tr_data,0) # size: n_class * n_bands
    training_cov[i,:,:] = np.cov(tr_data.T) # size: n_class * n_bands * n_bands
del tr_data

# Calculating discriminant function and then classified image
# Unlike NB in sklearn we consider covariance between dimensions instead of variance
g_fun = np.zeros((n, n_cl))
y_pred = np.zeros((n))
for x in range(n):
    for i in range(n_cl):
        mat = data[x,:] - training_mean[i,:]
        g_fun[x,i] = -0.5 * mat .dot(np.linalg.inv(training_cov[i,:,:])) .dot(
            mat.T) - 0.5 * np.log(np.linalg.det(training_cov[i,:,:])) + Pc0[0,i]
    y_pred[x] = labels[np.argmax(g_fun[x,:])]

del mat, g_fun

# Plot classified image
plt.figure(3) 
plt.imshow(y_pred.reshape((r,c)))

# Validation:
class_acc = [0]*n_cl
n_classified = []
n_validate = []
for i in range(n_cl):
    n_classified.append( len(np.where((y_pred==labels[i]) & (
        y_te==labels[i]))[0]) )
    n_validate.append( len(np.where( y_te==labels[i])[0]) )
    class_acc[i] = n_classified[i]/n_validate[i] * 100 

overal_acc = sum(n_classified)/sum(n_validate) * 100

print('Overal accuracy is: ', overal_acc)
for i,j in enumerate(class_acc):
    print("Accuracy of class {} is {}".format(i+1,j))
