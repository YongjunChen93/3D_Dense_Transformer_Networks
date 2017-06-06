#python 3.0
#coding:UTF-8
'''
@author Yongjun Chen 06/05/2017
'''
#for  set up the dataset

import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pylab import *


def show_data(data,label):
	[d1,d2,d3,d4] = data.shape
	for slices in range(d1):
		for depth in range(d2):
			#gray()
			plt.figure(figsize=(8,7),dpi=98)
			p1 = plt.subplot(211)
			p1.imshow(data[slices,depth,:,:])
			title_data = 'data batch:' + str(slices+1) + 'th' + ' slices: ' + str(depth+1)
			plt.title(title_data)
			p2 = plt.subplot(212)
			p2.imshow(label[slices,depth,:,:])
			title_label =  'label batch:' + str(slices+1) + 'th' + ' slices: ' + str(depth+1)
			plt.title(title_label)
			plt.pause(0.000001)
			plt.close()

def save(data,label,dataFileresult):
	[d1,d2,d3,d4] = data.shape
	result = h5py.File(dataFileresult,'w')
	result.create_dataset('Data', (d1,d2,d3,d4), dtype='float32')
	result.create_dataset('Label', (d1,d2,d3,d4), dtype='uint8')
	result['data'] = data
	result['label'] = label
	result.close()

def read_data(dataFile,size,Transpose=False,resize=False):
	f            =      h5py.File(dataFile,'r')
	if Transpose == True:
		data     =      np.transpose(np.array(f['data']),(3,2,1,0))
		label    =      np.transpose(np.array(f['label']),(3,2,1,0))
	else:
		data     =      np.array(f['data'])
		label    =      np.array(f['label'])
	if resize == True:
		data     =      np.resize(data,size)
		label    =      np.resize(label,size)
	return data, label