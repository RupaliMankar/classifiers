# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 13:04:30 2017

@author: Rupali
"""


import envi

#machine learning library
#import sklearn.ensemble      # for random forest

#import svm classifier
from sklearn import svm

#glob allows you to get file names with wildcards
import glob

#classification tools 
import classify

#display functions
import matplotlib.pyplot as plt

#load the scipy library
import scipy
import scipy.misc
import numpy


#load the mask used to interact with the ENVI file
mask = scipy.misc.imread("D:/breast-hm/train/mask.bmp", flatten=True).astype(numpy.bool)

#create and open an ENVI file
E = envi.envi("D:/breast-hm/train/1-norm-bip", mask=mask)

#get a set of class file names
classfiles = glob.glob("D:/breast-hm/train/train-resample/*.bmp")

#create a class image from a list of file names
C = classify.filenames2class(classfiles)

#load a training set from the ENVI file
Ft, Tt = E.loadtrain(C)


#create a classifier
clf =  sklearn.ensemble.RandomForestClassifier(n_estimators=100)     #randomforest classifier
#clf = svm.svc()

#train the classifier
clf.fit(Ft, Tt)


#load the mask used to interact with the ENVI file
mask = scipy.misc.imread("mask-8.bmp", flatten=True).astype(numpy.bool)

#create and open an ENVI file
E = envi.envi("D:/breast-hm/train/1-norm-bip", mask=mask)

#validate the classifier on the ENVI file
RGB = classify.envi_batch_predict(E, clf, 10000)

plt.imsave('rf-no-prep.png', RGB)
#close the ENVI file
E.close()
