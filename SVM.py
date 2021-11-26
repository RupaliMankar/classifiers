# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:19:27 2020

@author: Rupali
"""

import numpy as np
from PIL import Image
from sklearn import svm
import envi
import os
#classification tools 
import classify
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def maskreconstruct(maskpath, tissuemaskfile, Tpred, numberofclasses):  #this function converts 1D prediction values in 2D image
    os.chdir(maskpath)
    mask = plt.imread(tissuemaskfile)        
    X_loc = np.transpose(np.nonzero(mask[:,:,0]))
    cls2d = np.zeros(mask.shape[:-1])
    if(X_loc.shape[0]!=len(Tpred)):
        print("check all file names something is wrong")
    else:
       
       # C = np.zeros((numberofclasses, mask.shape[0],mask.shape[1]))
        for i in range(X_loc.shape[0]):
            cls2d[X_loc[i,0],X_loc[i,1]] = Tpred[i]
         #   C[int(Tpred[i]-1),X_loc[i,0],X_loc[i,1]] = 1
    
   # classifiedImg = classify.class2color(C)
    return cls2d#, classifiedImg

def IdentifyFalsePred(maskfiles,classImg2d):   #find incorrectly classsified pixels for each class
    falsePred = np.zeros((len(maskfiles),classImg2d.shape[0],classImg2d.shape[1]))
    i = 0
    for maskfile in maskfiles:
        mask = np.array(Image.open(maskfile.replace("#", str(tiles[0]))).convert("L"),np.bool)
        falsePred[i,:,:] = ((mask==1)&(classImg2d!=i+1))
        i = i+1
    return falsePred


#load training data and train the classifier 

#set file paths 
maskpath = "T:/Rupali/pansharpening/OVARY"  #maskpath 
ftirpath = "T:/Rupali/pansharpening/OVARY" #EnvifilePath

tiles = ["c2","c4"] # list of cores
maskfiles = ["#/cl-epith.bmp","#/cl-stroma.bmp"] #list of classes for raw
ftirfile =  "#/rowc-b-#-up-select-n-bip" #raw
os.chdir(ftirpath) 
#create and open an ENVI file
E = envi.envi(ftirfile.replace("#", tiles[0]))
Ft = np.empty((0, E.header.bands))
Tt = np.empty((0))
for tile in tiles:
    #go to data directory 
    os.chdir(ftirpath) 

    #create and open an ENVI file
    E = envi.envi(ftirfile.replace("#", tile))

    classfiles = []
    #get a set of class file names
    for maskfile in maskfiles:
        classfiles.append(maskfile.replace("#",tile))

    #create a class image from a list of file names
    C = classify.filenames2class(classfiles)

    #load a training set from the ENVI file
    F, T = E.loadtrain(C)
    Ft = np.concatenate((Ft,F),axis=0)
    Tt = np.concatenate((Tt,T),axis=0)

print('data has been loaded')


#train the classifier
# rbf_svc = svm.SVC(kernel='rbf')
# rbf_svc.fit(Ft,Tt)
clf = svm.SVC()
clf.fit(Ft, Tt)


#load data for testing 
tiles = ["c3"] # list of cores
maskfiles = ["#/mask-up.bmp"] #list of classes for raw
ftirfile =  "#/rowc-b-#-up-select-n-bip" #raw
os.chdir(ftirpath) 
#create and open an ENVI file
E = envi.envi(ftirfile.replace("#", tiles[0]))
Fv = np.empty((0, E.header.bands))
Tv = np.empty((0))
for tile in tiles:
    #go to data directory 
    os.chdir(ftirpath) 

    #create and open an ENVI file
    E = envi.envi(ftirfile.replace("#", tile))

    classfiles = []
    #get a set of class file names
    for maskfile in maskfiles:
        classfiles.append(maskfile.replace("#",tile))

    #create a class image from a list of file names
    C = classify.filenames2class(classfiles)

    #load a training set from the ENVI file
    F, T = E.loadtrain(C)
    Fv = np.concatenate((Fv,F),axis=0)
    Tv = np.concatenate((Tv,T),axis=0)
E.close()



#Quantitative and qualitstive results of classification
#Tpred = rbf_svc.predict(Fv)
Tpred = clf.predict(Fv)

# #T_pred = tt
# C = confusion_matrix(Tv,Tpred)
# Acc = accuracy_score(Tv, Tpred)

#classification of entire tissue (change classfile to tissuemask in validation code and change 1d predictions to 2d here)
tissuemaskfile = "c3/mask-up.bmp"
classImg2d = maskreconstruct(maskpath, tissuemaskfile,Tpred,2) #convert 1D classification into C*Y*X tensor
falsepred = IdentifyFalsePred(maskfiles,classImg2d) #Identify wrongly classified pixels for each class


