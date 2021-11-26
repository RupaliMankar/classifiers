# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 20:18:01 2017

@author: david
"""

import os
import numpy
import scipy
import matplotlib.pyplot as plt
import sys
from math import floor
#import progressbar2

class envi_header:
    def __init__(self, filename = ""):
        if filename != "":
            self.initialize()
            self.load(filename)
        else:
            self.initialize()
        
    #initialization function
    def initialize(self):
        self.samples = int(0)
        self.lines = int(0)
        self.bands = int(0)
        self.header_offset = int(0)
        self.data_type = int(4)
        self.interleave = "bsq"
        self.sensor_type = "Unknown"
        self.byte_order = int(0)
        self.x_start = int(0)
        self.y_start = int(0)
        self.z_plot_titles = "Unknown, Unknown"
        self.pixel_size = [float(0), float(0)]
        self.pixel_size_units = "Meters"
        self.wavelength_units = "Wavenumber"
        self.description = "no description"
        self.band_names = []
        self.wavelength = []
        
    #convert an ENVI data_type value to a numpy data type        
    def get_numpy_type(self, val):
        if val == 1:
            return numpy.byte
        elif val == 2:
            return numpy.int16
        elif val == 3:
            return numpy.int32
        elif val == 4:
            return numpy.float32
        elif val == 5:
            return numpy.float64
        elif val == 6:
            return numpy.complex64
        elif val == 9:
            return numpy.complex128
        elif val == 12:
            return numpy.uint16
        elif val == 13:
            return numpy.uint32
        elif val == 14:
            return numpy.int64
        elif val == 15:
            return numpy.uint64
    
    def get_envi_type(self, val):
        if val == numpy.byte:
            return 1
        elif val == numpy.int16:
            return 2
        elif val == numpy.int32:
            return 3
        elif val == numpy.float32:
            return 4
        elif val == numpy.float64:
            return 5
        elif val == numpy.complex64:
            return 6
        elif val == numpy.complex128:
            return 9
        elif val == numpy.uint16:
            return 12
        elif val == numpy.uint32:
            return 13
        elif val == numpy.int64:
            return 14
        elif val == numpy.uint64:
            return 15
            
    def load(self, fname):
        f = open(fname)
        l = f.readlines()
        if l[0].strip() != "ENVI":
            print("ERROR: not an ENVI file")
            return
        li = 1
        while li < len(l):
            #t = l[li].split()               #split the line into tokens
            #t = map(str.strip, t)               #strip all of the tokens in the token list
            
            #handle the simple conditions
            #if l[li].startswith("file type"):
            #    if not l[li].strip().endswith("ENVI Standard"):
            #        print("ERROR: unsupported ENVI file format: " + l[li].strip())
            #        return
            if l[li].startswith("samples"):
                self.samples = int(l[li].split()[-1])
            elif l[li].startswith("lines"):
                self.lines = int(l[li].split()[-1])
            elif l[li].startswith("bands"):
                self.bands = int(l[li].split()[-1])
            elif l[li].startswith("header offset"):
                self.header_offset = int(l[li].split()[-1])
            elif l[li].startswith("data type"):
                self.data_type = self.get_numpy_type(int(l[li].split()[-1]))
            elif l[li].startswith("interleave"):
                self.interleave = l[li].split()[-1].strip()
            elif l[li].startswith("sensor type"):
                self.sensor_type = l[li].split()[-1].strip()
            elif l[li].startswith("byte order"):
                self.byte_order = int(l[li].split()[-1])
            elif l[li].startswith("x start"):
                self.x_start = int(l[li].split()[-1])
            elif l[li].startswith("y start"):
                self.y_start = int(l[li].split()[-1])
            elif l[li].startswith("z plot titles"):
                i0 = l[li].rindex('{')
                i1 = l[li].rindex('}')
                self.z_plot_titles = l[li][i0 + 1 : i1]
            elif l[li].startswith("pixel size"):
                i0 = l[li].rindex('{')
                i1 = l[li].rindex('}')
                s = l[li][i0 + 1 : i1].split(',')
                self.pixel_size = [float(s[0]), float(s[1])]
                self.pixel_size_units = s[2][s[2].rindex('=') + 1:].strip()
            elif l[li].startswith("wavelength units"):
                self.wavelength_units = l[li].split()[-1].strip()                
            
            #handle the complicated conditions
            elif l[li].startswith("description"):
                desc = [l[li]]
                ''' 
                while l[li].strip()[-1] != '}': #will fail if l[li].strip() is empty
                    li += 1
                    desc.append(l[li])
                '''
                while True:
                    if l[li].strip():
                       if  l[li].strip()[-1] == '}':
                           break
                    li += 1
                    desc.append(l[li])

                desc = ''.join(list(map(str.strip, desc)))           #strip all white space from the string list
                i0 = desc.rindex('{')
                i1 = desc.rindex('}')
                self.description = desc[i0 + 1 : i1]
                
            elif l[li].startswith("band names"):
                names = [l[li]]
                while l[li].strip()[-1] != '}':
                    li += 1
                    names.append(l[li])
                names = ''.join(list(map(str.strip, names)))           #strip all white space from the string list
                i0 = names.rindex('{')
                i1 = names.rindex('}')
                names = names[i0 + 1 : i1]
                self.band_names = list(map(str.strip, names.split(',')))
            elif l[li].startswith("wavelength"):
                waves = [l[li]]
                while l[li].strip()[-1] != '}':
                    li += 1
                    waves.append(l[li])
                waves = ''.join(list(map(str.strip, waves)))           #strip all white space from the string list
                i0 = waves.rindex('{')
                i1 = waves.rindex('}')
                waves = waves[i0 + 1 : i1]
                self.wavelength = list(map(float, waves.split(',')))

            li += 1          
        
        f.close()

    #save an ENVI header
    def save(self, fname):
        f = open(fname, "w")
        f.write("ENVI\n")
        f.write("description = {" + self.description + "}" + "\n")
        f.write("samples = " + str(self.samples) + "\n")
        f.write("lines = " + str(self.lines) + "\n")
        f.write("bands = " + str(self.bands) + "\n")
        f.write("header offset = " + str(self.header_offset) + "\n")
        f.write("file type = ENVI Standard" + "\n")
        f.write("data type = " + str(self.get_envi_type(self.data_type)) + "\n")
        f.write("interleave = " + self.interleave + "\n")
        f.write("sensor type = " + self.sensor_type + "\n")
        f.write("byte order = " + str(self.byte_order) + "\n")
        f.write("x start = " + str(self.x_start) + "\n")
        f.write("y start = " + str(self.y_start) + "\n")
        f.write("wavelength units = " + self.wavelength_units + "\n")
        f.write("z plot titles = {" + self.z_plot_titles + "}" + "\n")
        
        # save the wavelength values
        if self.wavelength != []:
            if len(self.wavelength) == self.bands:
                f.write("wavelength = {")
                f.write(",".join(map(str, self.wavelength)))
                f.write("}\n")
            else:
                raise Exception("ENVI HEADER ERROR: Number of wavelengths does not match number of bands")

        f.close()

    #sets the properties of the header to match those of the input array
    def setprops(self, A, interleave="BSQ", wavelength=[]):
        # determine the data type automatically
        self.data_type = A.dtype
        
        # determine the ordering based on the specified interleave
        if interleave == "BSQ":
            self.samples = A.shape[2]
            self.lines = A.shape[1]
            self.bands = A.shape[0]
        elif interleave == "BIP":
            self.samples = A.shape[1]
            self.lines = A.shape[2]
            self.bands = A.shape[0]
        elif interleave == "BIL":
            self.samples = A.shape[0]
            self.lines = A.shape[2]
            self.bands = A.shape[1]
        else:
            raise Exception("invalid interleave format (requires 'BSQ', 'BIP', or 'BIL') - interleave is set to {}".interleave)
            
        # if wavelength units are given, make sure that they match the number of bands
        if wavelength != []:
            if len(wavelength) != self.bands:
                raise Exception("invalid number of wavelengths specified")
            else:
                self.wavelength = wavelength
                

        
class envi:
    def __init__(self, filename, headername = "", mask = []):
        self.open(filename, headername)
        if mask == []:
            self.mask = numpy.ones((self.header.lines, self.header.samples), dtype=numpy.bool)
        elif type(mask) == numpy.ndarray:
            self.mask = mask
        else:
            print("ERROR: unrecognized mask format - expecting a boolean array")
        self.idx = 0                                                               #initialize the batch IDX to 0 for batch reading
        
    def open(self, filename, headername = ""):
        if headername == "":
            headername = filename + ".hdr"
            
        if not os.path.isfile(filename):
            print("ERROR: " + filename + " not found")
            return
        if not os.path.isfile(headername):
            print("ERROR: " + headername + " not found")
            return
        
        #open the file
        self.header = envi_header(headername)
        self.file = open(filename, "rb")
    
    # load the entire ENVI file into memory and return it as an array
    def loadall(self):
        X = self.header.samples
        Y = self.header.lines
        B = self.header.bands
        
        #load the data
        D = numpy.fromfile(self.file, dtype=self.header.data_type)
        
        if self.header.interleave == "bsq":
            return numpy.reshape(D, (B, Y, X))
            #return numpy.swapaxes(D, 0, 2)
        elif self.header.interleave == "bip":
            D = numpy.reshape(D, (Y, X, B))
            return numpy.rollaxis(D, 2)
        elif self.header.interleave == "bil":
            D = numpy.reshape(D, (Y, B, X))
            return numpy.rollaxis(D, 1)
    
    #save an updated version of the file (all header information is assumed to be the same)
    def saveall(self, D, filename):
        
        new_header = self.header
        new_header.interleave = "bsq"
        new_header.save(filename + ".hdr")
        D.tofile(filename)
        
        
    #loads all of the pixels where mask != 0 and returns them as a matrix
    def loadmask(self, mask):
        X = self.header.samples
        Y = self.header.lines
        B = self.header.bands
        
        P = numpy.count_nonzero(mask)           #count the number of zeros in the mask file
        M = numpy.zeros((B, P), dtype=self.header.data_type)
        type_bytes = numpy.dtype(self.header.data_type).itemsize
        
        prev_pos = self.file.tell()
        self.file.seek(0)
        if self.header.interleave == "bip":
            spectrum = numpy.zeros(B, dtype=self.header.data_type)
            flatmask = numpy.reshape(mask, (X * Y))
            i = numpy.flatnonzero(flatmask)
            #bar = progressbar2.ProgressBar(max_value = P)
            #bar = pyprind.ProgBar(P)
            for p in range(0, P):
                self.file.seek(i[p] * B * type_bytes)
                self.file.readinto(spectrum)
                M[:, p] = spectrum
                #bar.update(p+1)
                #bar.update()
        elif self.header.interleave == "bsq":
            band = numpy.zeros(mask.shape, dtype=self.header.data_type)
            i = numpy.nonzero(mask)
            #bar = progressbar2.ProgressBar(max_value=B)
            #bar = pyprind.ProgBar(P)
            for b in range(0, B):
                self.file.seek(b * X * Y * type_bytes)
                self.file.readinto(band)
                M[b, :] = band[i]
                #bar.update(b+1)
                #bar.update()
        elif self.header.interleave == "bil":
            plane = numpy.zeros((B, X), dtype=self.header.data_type)
            p = 0
            #bar = progressbar2.ProgressBar(max_value=Y)
            #bar = pyprind.ProgBar(P)
            for l in range(0, Y):
                i = numpy.flatnonzero(mask[l, :])
                self.file.readinto(plane)
                M[:, p:p+i.shape[0]] = plane[:, i]
                p = p + i.shape[0]
                #bar.update(l+1)
                #bar.update()
        self.file.seek(prev_pos)
        return M

    def loadband(self, n):
        X = self.header.samples
        Y = self.header.lines
        B = self.header.bands

        band = numpy.zeros((Y, X), dtype=self.header.data_type)
        type_bytes = numpy.dtype(self.header.data_type).itemsize
        
        prev_pos = self.file.tell()
        if self.header.interleave == "bsq":
            self.file.seek(n * X * Y * type_bytes)
            self.file.readinto(band)
        self.file.seek(prev_pos)
        return band

    #create a set of feature/target pairs for classification
    #input: envi file object, stack of class masks C x Y x X
    #output: feature matrix (features x pixels), target matrix (1 x pixels)
    #example: generate_training(("class_coll.bmp", "class_epith.bmp"), (1, 2))
    #   verify      verify that there are no NaN or Inf values
    def loadtrain(self, classimages, verify=True):

        # get number of classes
        C = classimages.shape[0]

        F = []
        T = []
        for c in range(0, C):
            print("\nLoading class " + str(c+1) + "...")
            f = self.loadmask(classimages[c, :, :])            #load the feature matrix for class c
            t = numpy.ones((f.shape[1])) * (c+1)         #generate a target array                 
            F.append(f)
            T.append(t)
        
        return numpy.nan_to_num(numpy.concatenate(F, 1).transpose()), numpy.concatenate(T)


    #create a set of feature/target pairs for classification with balanced data
    #input: envi file object, stack of class masks C x Y x X, number of samples per class
    #output: feature matrix (features x pixels), target matrix (1 x pixels)
    #example: generate_training(("class_coll.bmp", "class_epith.bmp"), (1, 2))
    #   verify      verify that there are no NaN or Inf values
    def loadtrain_balance(self, classimages, num_samples=None):

        # get number of classes
        C = classimages.shape[0]

        F = []
        T = []

        # get number of samples per class
        samples_per_class = numpy.zeros(C, dtype=numpy.int32)
        for c in range(0, C):
            if num_samples is None:
                samples_per_class[c] = numpy.count_nonzero(classimages[c, :, :])
            else:
                # if user has specified a max number of samples per class
                if num_samples > numpy.count_nonzero(classimages[c, :, :]):
                    samples_per_class[c] = numpy.count_nonzero(classimages[c, :, :])
                else:
                    samples_per_class[c] = num_samples

        for c in range(0, C):
            print("\nLoading class " + str(c+1) + "...")
            # row, col index of valid pixels
            temp = classimages[c,:]
            flat_temp = numpy.reshape(temp, temp.shape[0]*temp.shape[1])

            idx = numpy.flatnonzero(temp)  # indices of nonzero values
            if num_samples:
                # use specific number of samples for training
                numpy.random.shuffle(idx)
                idx = idx[0:samples_per_class[c]]

            # increase number of samples by copying them over multiple times
            max_samples = numpy.amax(samples_per_class)
            # num of times to copy for even division
            copy_times = int(floor(max_samples / samples_per_class[c]))
            rem = max_samples % samples_per_class[c]  # remaining samples

            for i in range(0, copy_times):
                numpy.random.shuffle(idx)
                shuffle_temp = numpy.zeros(flat_temp.shape, dtype=bool)
                shuffle_temp[idx] = flat_temp[idx]
                f = self.loadmask(numpy.reshape(shuffle_temp, (temp.shape[0], temp.shape[1])))  # load the feature matrix for class c
                t = numpy.ones((f.shape[1])) * (c+1)  # generate a target array
                F.append(f)
                T.append(t)

            # copy the remaning samples so the total matches the max number of samples chosen by user
            if rem > 0:
                numpy.random.shuffle(idx)
                idx = idx[0:rem]
                shuffle_temp = numpy.zeros(flat_temp.shape, dtype=bool)
                shuffle_temp[idx] = flat_temp[idx]
                f = self.loadmask(numpy.reshape(shuffle_temp, (temp.shape[0], temp.shape[1])))  # load the feature matrix for class c
                t = numpy.ones((f.shape[1])) * (c+1)  # generate a target array
                F.append(f)
                T.append(t)

        return numpy.nan_to_num(numpy.concatenate(F, 1).transpose()), numpy.concatenate(T)

    
    #read a batch of data based on the mask
    def loadbatch(self, npixels):
        i = numpy.flatnonzero(self.mask)                                      #get the indices of valid pixels
        if len(i) == self.idx:                                                    #if all of the pixels have been read, return an empyt array
            return []
        npixels = min(npixels, len(i) - self.idx)                        #if there aren't enough pixels, change the batch size
        B = self.header.bands
        
        batch = numpy.zeros((B, npixels), dtype=self.header.data_type)          #allocate space for the batch
        pixel = numpy.zeros((B), dtype=self.header.data_type)                   #allocate space for a single pixel
        type_bytes = numpy.dtype(self.header.data_type).itemsize                #calculate the size of a single value
        if self.header.interleave == "bip":
            for n in range(0, npixels):                                          #for each pixel in the batch
                self.file.seek(i[self.idx] * B * type_bytes)                 #seek to the current pixel in the file
                self.file.readinto(pixel)                                       #read a single pixel
                batch[:, n] = pixel                                             #save the pixel into the batch matrix
                self.idx = self.idx + 1
            return batch
        elif self.header.interleave == "bsq":
            print("ERROR: BSQ batch loading isn't implemented yet!")
        elif self.header.interleave == "bil":
            print("ERROR: BIL batch loading isn't implemented yet!")        
       
    #returns the current batch index         
    def getidx(self):
        return self.idx

    #returns an image of the pixels that have been read using batch loading
    def batchmask(self):
        #allocate a new mask
        outmask = numpy.zeros(self.mask.shape, dtype=numpy.bool)

        #zero out any unclassified pixels 
        idx = self.getidx()
        i = numpy.nonzero(self.mask)
        outmask[i[0][0:idx], i[1][0:idx]] = self.mask[i[0][0:idx], i[1][0:idx]]
        return outmask

    def close(self):
        self.file.close()
            
    def __del__(self):
        self.file.close()

#saves an array as an ENVI file
def save_envi(A, fname, interleave="BSQ", wavelength=[]):
    
    #create and save a header file
    header = envi_header();
    header.setprops(A, interleave, wavelength)
    header.save(fname + ".hdr")

    #save the raw data
    file = open(fname, "wb")
    file.write(bytearray(A))
    file.close()
