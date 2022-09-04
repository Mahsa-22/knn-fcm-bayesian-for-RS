# knn-fcm-bayesian-for-RS
Some basic ML python code in RS
These codes are for remote sensing and machine learning beginers to understand how
 to implement ML algorithms in python.
To understand these codes you need to have basic knowledge of ML, RS science, python,
 numpy and matplotlib.

knn classification:
    inputs: 1- Satellite image of size d*r*c (d is the number of sat image bands)
            2- Train image of size r*c (some pixels are labled and the rest are zero)
            3- Test image of size r*c (some pixels are labled and the rest are zero)
               (Labels are obtained from the visual interpretation of the sat image)
            4- Range of k numbers
    output: finding best k for classification and overal accuracies
            a plot of classified image

Procedural fcm:
    inputs: 1- satellite image of size d*r*c (d is the number of sat image bands)
            2- some initial values: (cl: number of clusters, q: fuzzifier, lmax: max
               number of iterations)
    output: a plot of clustered image with title and legend

Object based fcm:
    inputs: 1- satellite image of size d*r*c (d is the number of sat image bands)
            2- some initial values: (cl: number of clusters, q: fuzzifier, lmax: max
               number of iterations)
    output: a plot of clustered image

Bayesian classification:
    inputs: 1- Satellite image of size d*r*c (d is the number of sat image bands)
            2- Target image of size r*c (some pixels are labled and the rest are zero)
               (Labels are obtained from the visual interpretation of the sat image)
    output: Overal accuracy and class accuracies
            and plot of train and test label images and classified image
