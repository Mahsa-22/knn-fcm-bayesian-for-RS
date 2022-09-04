These codes are for remote sensing and machine learning beginners to understand how to implement ML algorithms in python. To understand these codes you need to have basic knowledge of ML, RS science, python, numpy and matplotlib.

------------
**knn classification:**
- Inputs:
   1. Satellite image of size d*r*c (d is the number of sat image bands)
   1. Train image of size r*c (some pixels are labeled and the rest are zero)
   1. Test image of size r*c (some pixels are labeled and the rest are zero)
   (Labels are obtained from the visual interpretation of the sat image)
   1. Range of k numbers
- Output:
   1. Best k for classification and overall accuracies and a plot of classified image

------------
**Procedural FCM:**
- Inputs:
   1. Satellite image of size d*r*c (d is the number of sat image bands)
   1. Some initial values: (cl: number of clusters, q: fuzzifier, lmax: max number of iterations)
- Output:
   1. Clustered image and its plot with title and legend

------------
**Object based FCM:**
- Inputs:
   1. Satellite image of size d*r*c (d is the number of sat image bands)
   1. Some initial values: (cl: number of clusters, q: fuzzifier, lmax: max
number of iterations)
- Output:
   1. Clustered image and its plot

------------
**Bayesian classification:**
- Inputs:
   1. Satellite image of size d*r*c (d is the number of sat image bands)
   1. Target image of size r*c (some pixels are labeled and the rest are zero) (Labels are obtained from the visual interpretation of the sat image)
- Output:
   1. Overall accuracy and class accuracies and plots of train and test label images and classified image
