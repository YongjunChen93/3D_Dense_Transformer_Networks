#python 3.0
#coding:UTF-8
'''
@author Yongjun Chen 06/01/2017
'''
#for  set up the dataset

import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pylab import *
import tensorflow as tf


def augmentation(data,label,flip_up_down,flip_left_right,rot90,dir_loc,loc):
	[d1,d2,d3,d4]          =          data.shape
	filename               =          dir_loc + loc + 'LON_'+str(flip_up_down)+str(flip_left_right)+str(rot90)+'.h5'
	aug_file               =          h5py.File(filename,'w')

	for batch_index in range(d1):
		for index in range(d2):
			if flip_up_down == 1:
				data[batch_index][index]    =     np.flipud(data[batch_index][index])
				label[batch_index][index]   =     np.flipud(label[batch_index][index])
			if flip_left_right == 1:
				data[batch_index][index]    =     np.fliplr(data[batch_index][index])
				label[batch_index][index]   =     np.fliplr(label[batch_index][index])
			if rot90 == 1:
				data[batch_index][index]    =     np.rot90(data[batch_index][index])
				label[batch_index][index]   =     np.rot90(label[batch_index][index])
	for batch_index in range(d1):
		for h in range(d3):
			if flip_up_down == 2:
				data[batch_index,:,h,:]  =     np.flipud(data[batch_index,:,h,:])
				label[batch_index,:,h,:] =     np.flipud(label[batch_index,:,h,:])
			if flip_left_right == 2:

				data[batch_index,:,h,:]  =     np.fliplr(data[batch_index,:,h,:])
				label[batch_index,:,h,:]  =     np.fliplr(label[batch_index,:,h,:])
			if rot90 == 2:
				data[batch_index][index]    =     np.rot90(np.rot90(data[batch_index][index]))
				label[batch_index][index]   =     np.rot90(np.rot90(label[batch_index][index]))
	for batch_index in range(d1):
		for w in range(d4):
			if flip_up_down == 3:
				data[batch_index,:,:,w]   =     np.flipud(data[batch_index,:,:,w])
				label[batch_index,:,:,w]  =     np.flipud(label[batch_index,:,:,w])						
			if flip_left_right == 3:
				data[batch_index,:,:,w]   =     np.fliplr(data[batch_index,:,:,w])
				label[batch_index,:,:,w]  =     np.fliplr(label[batch_index,:,:,w])			
			if rot90 == 3:
				data[batch_index][index]    =     np.rot90(np.rot90(np.rot90(data[batch_index][index])))
				label[batch_index][index]   =     np.rot90(np.rot90(np.rot90(label[batch_index][index])))						
	aug_file.create_dataset('data', data = data)
	aug_file.create_dataset('label', data = label)
	aug_file.close()

def do_augmentation(file,dir_loc,loc):
	data     =      np.array(file['data'])
	label    =      np.array(file['label'])
	[d1,d2,d3,d4]     =     data.shape
	for flipud in range(4):
		for fliplr in range(4):
			for rotn in range(1):
				print("flipud",flipud,"fliplr",fliplr,"rotn",rotn)
				augmentation(data,label,flipud,fliplr,rotn,dir_loc,loc)

def main():	
	dir_loc      =       '../augment_data/'
	train_loc    =		 'LONI_train/'
	valid_loc    =     	 'LONI_valid/'
	test_loc     =       'LONI_test/'
	train_save_loc = 'LONI_train.h5'
	valid_save_loc = 'LONI_valid.h5'
	test_save_loc  = 'LONI_test.h5'
	print("start train")
	f        =      h5py.File(train_save_loc)
	do_augmentation(f,dir_loc,train_loc)
	print("start valid")
	f        =      h5py.File(valid_save_loc)
	do_augmentation(f,dir_loc,valid_loc)
	print("start test")
	f        =      h5py.File(test_save_loc)
	do_augmentation(f,dir_loc,test_loc)

if __name__ == '__main__':
	main()





