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


def augmentation(data,label,flip_up_down,flip_left_right,rot90,loc):
	[d1,d2,d3,d4]          =          data.shape
	dir_loc                =          'augment_data/'
	filename               =          dir_loc + loc + 'LON_'+str(flip_up_down)+str(flip_left_right)+str(rot90)+'.h5'
	aug_file               =          h5py.File(filename,'w')
	aug_file.create_dataset('Data', (d1,d2,d3,d4), dtype='uint8')
	aug_file.create_dataset('Label', (d1,d2,d3,d4), dtype='uint8')

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
	aug_file['data']       =          data
	aug_file['label']      =          label
	aug_file.close()

def do_augmentation(file,loc):
	data     =      np.array(f['data'])
	label    =      np.array(f['label'])
	[d1,d2,d3,d4]     =     data.shape
	for flipud in range(4):
		for fliplr in range(4):
			for rotn in range(4):
				print("flipud",flipud,"fliplr",fliplr,"rotn",rotn)
				augmentation(data,label,flipud,fliplr,rotn,loc)

def main():
	print("start all")				
	dataFile = './h5data/LONI_cut_margin_all.h5'
	f        =      h5py.File(dataFile)
	do_augmentation(f,'LONI_All/')
	print("start train")
	dataFile = './h5data/LONI_cut_margin_train.h5'
	f        =      h5py.File(dataFile)
	do_augmentation(f,'LONI_train/')
	print("start valid")
	dataFile = './h5data/LONI_cut_margin_valid.h5'
	f        =      h5py.File(dataFile)
	do_augmentation(f,'LONI_valid/')
	print("start test")
	dataFile = './h5data/LONI_cut_margin_test.h5'
	f        =      h5py.File(dataFile)
	do_augmentation(f,'LONI_test/')

if __name__ == '__main__':
	main()





