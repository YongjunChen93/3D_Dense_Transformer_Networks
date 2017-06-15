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
import os

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
	print("test",type(data))
	[d1,d2,d3,d4] = data.shape
	result = h5py.File(dataFileresult,'w')
	result.create_dataset('data', data = data)
	result.create_dataset('label', data = label)
	result.close()

def read_data(dataFile,size,Transpose=False,resize=False):
	f             =      h5py.File(dataFile,'r')
	result_data   =      np.zeros(size)
	result_label  =      np.zeros(size)
	if Transpose == True:
		data      =      np.transpose(np.array(f['data']),(3,2,1,0))
		label     =      np.transpose(np.array(f['label']),(3,2,1,0))
	else:
		data      =      np.array(f['data'])
		label     =      np.array(f['label'])
	[d1,d2,d3,d4] =      data.shape
	if resize == True:
		for p in range(d1):
			for d in range(d2):
				result_data[p,d,:,:]     =      np.resize(data[p,d,:,:],(size[2],size[3]))
				result_label[p,d,:,:]    =      np.resize(label[p,d,:,:],(size[2],size[3]))
		data      =      result_data
		label     =      result_label
	return data, label

def check_name(dataFile):
	f             =      h5py.File(dataFile,'r')
	for name in f:
		print("keys of f",name)

def make_data_loc_file(location,write_loc):
	filelist = os.listdir(location)
	with open(write_loc,"w") as f:
		for i in range(len(filelist)):
			f.write(location+'/'+filelist[i]+'\n')
	f.close()

if __name__ == '__main__':
	size = (1,2,3,4)
	print(size[1])
	make_data_loc_file('../augment_data/LONI_train','../augment_data/LONI_train/train.txt')