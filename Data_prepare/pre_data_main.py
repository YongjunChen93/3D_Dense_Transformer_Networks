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
from prepare_data import *

def main():
	dataFile =      'LONI_test.h5'
	dataFileresult = 'LONI_test.h5'
	print("dataFile1",dataFile)
	size = (10,181,150,150)
	data,label = read_data(dataFile,size,Transpose=False,resize=False)
	print(type(data[1,1,1,1]))
	print(data.shape)
	#save(data,label,dataFileresult)
	show_data(data,label)

if __name__ == '__main__':
	main()

