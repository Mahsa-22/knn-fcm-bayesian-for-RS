# -*- coding: utf-8 -*-
"""
Created on Jun 2022

@author: Mahsa Farahani
mahsafarahani971@gmail.com

FCM clustering for remote sensing images (object based)

"""

# import necessary libraries
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt

# input satellite image data
with rio.open("input your satellite image address here") as img:
    image_data = img.read()    

d, r, c = image_data.shape
n = r*c

# reshape data
data = np.zeros((n,d))
for i in range(d):
    data[:,i] = image_data[i,:,:].flatten()

del image_data

#####
class _FCM_():
    
    def __init__(self, cl, q = 1.5, lmax = 50): # you can change the default values for q and lmax
        self.cl = cl
        self.lmax = lmax
        self.q = q
        
    def fit_FCM_(self, x_train):
        self.X = x_train
    
    def predicted_FCM_(self):
        
        n, d = self.X.shape
        # Random membership values
        self.U = np.random.rand(n, self.cl) #initilizing memeberships
        self.U = self.U/(np.sum(self.U,1).reshape((n,1))) # rows summing will equal to 1
        self.V = np.zeros((d, self.cl))
        l = 0
        
        while l < self.lmax:
            
            l += 1
            # New cluster centroid
            uu = self.U**self.q
            for i in range(d):  
                self.V[i,:] = np.sum(uu * self.X[:,i].reshape((n,1)), 0) / np.sum(uu,0)
            
            # Similarity measure
            self.distance = np.zeros((n,self.cl))
            for i in range(self.cl):
                self.distance[:,i] = (np.sum((self.X - self.V[:,i])**2, 1))**0.5
            
            # New membership
            uu = np.zeros((n,self.cl))
            for i in range(self.cl):
                uu[:,i]=np.sum((((self.distance[:,i].reshape((n,1))) / self.distance)**(1/(self.q-1))),1)
            self.U = 1/uu
        return np.argmax(self.U,1)
    
    def _FCM_values(self):
        return self.U, self.V, self.distance
#####

# create fcm class object
img_cluster = _FCM_(6) # 6 is number of clusters (cl)
img_cluster.fit_FCM_(data) # fit to input data
clustered = img_cluster.predicted_FCM_() # implement fcm clustering

clustered_image = clustered.reshape((r,c)) # reshape n*1 to r*c

# plot clustered map (you can also use the method in "Procedural fcm" for plotting)
plt.figure()
plt.imshow(clustered_image)

# extract value of memberships (just for instance)
U_final = img_cluster._FCM_values()[0]
