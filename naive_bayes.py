# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:04:30 2017

@author: Rupali
"""
import envi

#machine learning library
import sklearn.naive_bayes

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
mask = scipy.misc.imread("mask.bmp", flatten=True).astype(numpy.bool)

#create and open an ENVI file
E = envi.envi("W:/berisha/brc961-hd/BRC961_Mosaic", mask=mask)

#get a set of class file names
classfiles = glob.glob("c_*.png")

#create a class image from a list of file names
C = classify.filenames2class(classfiles)
#

#load a training set from the ENVI file
Ft, Tt = E.loadtrain(C)


#create a classifier
CLASS = sklearn.naive_bayes.GaussianNB()

#train the classifier
CLASS.fit(Ft, Tt)
#load the mask used to interact with the ENVI file
mask = scipy.misc.imread("mask-4cores.png", flatten=True).astype(numpy.bool)

#create and open an ENVI file
E = envi.envi("D:/python-test/brc961-4cores-bip", mask=mask)

#validate the classifier on the ENVI file
classify.envi_batch_predict(E, CLASS)

#close the ENVI file
E.close()
