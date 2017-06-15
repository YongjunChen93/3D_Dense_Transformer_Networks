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
from prepare_data import *

def main():
	dataFile =      'LONI_valid.h5'
	dataFileresult = 'LONI_train.h5'
	print("dataFile1",dataFile)
	size = (10,181,142,149)
	check_name(dataFile)
	data,label = read_data(dataFile,size,Transpose=False,resize=False)
	save(data,label,dataFileresult)
	make_data_loc_file('../augment_data/LONI_train','../augment_data/LONI_train/train.txt')
	show_data(data,label)

if __name__ == '__main__':
	main()