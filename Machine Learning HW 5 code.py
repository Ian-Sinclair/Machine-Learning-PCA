# -*- coding: utf-8 -*-
"""
Created on Tue May 25 17:46:44 2021

@author: IanSi
"""

import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784',version=1,return_X_y=True)

j= 10 
plt.title('The jth image is a {label}'.format(label=int(y[j])))
plt.imshow(X[j].reshape((28,28)),cmap='gray')
plt.show()

X4 = X[y=='4',:]
X9 = X[y=='9',:]
y4 = 4*np.ones((len(X4),), dtype=int)
y9 = 9*np.ones((len(X9),), dtype=int)

X = np.concatenate((X4,X9),axis=0)
y = np.concatenate((y4,y9),axis=0)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.43,random_state=0)

#Preprocessing.
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)


"""
TASK 1: Selecting the number of components for PCA with 80% of Energy Retained.
"""

pca = PCA(n_components=784)
pca.fit(X_train)

Variance = pca.explained_variance_ratio_

SumEigenValue = 0

k = 0
for lamda in Variance:
    if SumEigenValue < 0.8:
        k += 1
        SumEigenValue += lamda


pca = PCA(n_components=k)
pca.fit(X_train)

Variance = pca.explained_variance_ratio_

#Total Vaiance = 0.80235

"""
TASK 2: Feature Reduction
"""

Theta_train = pca.transform(X_train)
Theta_test = pca.transform(X_test)


"""
Problem 1: Part B, inhomogeneous linear and quadratic kernel
"""

"""
Holdout Method
"""
#Partitions training data to sets used to train the classifier and test data for parameters.
Theta_fit,Theta_holdout,y_fit,y_holdout = train_test_split(Theta_train,y_train,test_size=0.3,random_state=0)

"""
Inhomogeneous Linear Kernel
"""

min_error = 1;
C_ = -1

#Select C
for n in range(-10,11,1):                          #Training on range [0.00006, 16384 : *2].
    clf = svm.SVC(C=2**n,kernel='poly',degree = 1) #Builds linear classifier
    clf.fit(Theta_fit,y_fit)                            #Trains classifier using the partitioned training data
    Pe = 1-clf.score(Theta_holdout,y_holdout)           #Tests classifier using the holdout set.
    if Pe < min_error:                                  #Maintains the C_ value corresponding to the minimum error.
        min_error = Pe
        C_ = 2**n

print(C_)
#Retrain
clf = svm.SVC(C=C_,kernel='poly',degree = 1)
clf.fit(Theta_train,y_train)
#Error and support vectors
Pe = 1-clf.score(Theta_test,y_test)
num_SV = clf.support_vectors_

#num_SV = 955, Pe = 0.03492491985827573, C_ = 2


"""
Inhomogeneous Quadratic Kernel
"""

min_error = 1;
C_ = -1

#Select C
for n in range(-20,21,1): 
    clf = svm.SVC(C=1.2**n,kernel='poly',degree = 2) 
    clf.fit(Theta_fit,y_fit) 
    Pe = 1-clf.score(Theta_holdout,y_holdout) 
    if Pe < min_error: 
        min_error = Pe
        C_ = 1.2**n

print(C_)
#Retrain
clf = svm.SVC(C=C_,kernel='poly',degree = 2)
clf.fit(Theta_train,y_train)
#Error and support vectors
Pe = 1-clf.score(Theta_test,y_test)
num_SV = clf.support_vectors_

#num_SV = 960, Pe = 0.014679, C_ = 8.9160045