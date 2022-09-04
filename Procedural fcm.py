# -*- coding: utf-8 -*-
"""
Created on Jun 2022

@author: Mahsa Farahani
mahsafarahani971@gmail.com

FCM (Fuzzy C-means) clustering for remote sensing images

"""

# import necessary libraries
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# input satellite image data
with rio.open("input your satellite image address here") as img:
    image_data = img.read()    

# shape and size 
d, r, c = image_data.shape
n = r*c

# reshape data (n*d)
data = np.zeros((n,d))
for i in range(d):
    data[:,i] = image_data[i,:,:].flatten()

del image_data

# primary value for parameters: number of clusters, fuzzifier q, max iteration
cl,q,lmax = 6,1.5,50

# initilizing memeberships
U = np.random.rand(n, cl)
U = U/(np.sum(U,1).reshape((n,1))) # rows summing will equal to 1
V = np.zeros((d, cl))

l = 0
while l < lmax:
    
    l += 1
    
    # new cluster centroid
    uu = U**q
    for i in range(d):
        V[i,:] = np.sum(uu * data[:,i].reshape((n,1)), 0) / np.sum(uu,0)

    # Similarity measure
    distance = np.zeros((n, cl))
    for i in range(cl):
        distance[:,i] = (np.sum(((data - V[:,i])**2), 1))**0.5
    
    # new memeberships
    uuu = np.zeros((n, cl))
    for i in range(cl):
        uuu[:,i] = np.sum((((distance[:,i].reshape((n,1))) / distance)**(1/(q-1))),1)
    U = 1/uuu
    print(l)

# clustered image            
cluster_vector = np.argmax(U,1)
cluster_image = cluster_vector.reshape((r,c))

## plot the clustered map
# you can creat a list of colors for your clustered map according to the
# number of clusters
fig, ax = plt.subplots()

colors = ['red', 'green', 'blue', 'cyan', 'yellow', 'magenta']
cmap = ListedColormap(colors)

class_idx = np.unique(cluster_vector)
legend_labels = dict(zip(colors,class_idx))
patches = [Patch(color=color, label=label) for color, label in legend_labels.items()]

ax.set_title("Clustered Map")
ax.legend(handles=patches, bbox_to_anchor=(1.25, 1))
ax.imshow(cluster_image, cmap = cmap)
plt.show()
