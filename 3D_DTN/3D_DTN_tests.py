# python3.0
# @ author Yongjun Chen
# Implement 3D version of Dense Transformer Layer
from Dense_Transformer_Networks_3D import *
import numpy as np
import tensorflow as tf
import h5py

def main():
	sess = tf.Session()
	# inputs
	U=tf.linspace(1.0,10.0,2*8*8*8*2)
	U =tf.reshape(U,[2,8,8,8,2])
	#network initial
	dtn_input_shape = [2,8,8,8,2]
	control_points_ratio = 2
	# initial DTN class
	transform = DSN_Transformer_3D(dtn_input_shape,control_points_ratio)
	# encoder
	conv1= transform.Encoder(U,U)
	#decoder
	conv2 = transform.Decoder(conv1,conv1)

if __name__ == "__main__":
    main()

