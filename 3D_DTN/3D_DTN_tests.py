# python3.0
# Implement 3D version of Dense Transformer Layer
from TPS_transformer import *
from SpatialDecoderLayer import *
import numpy as np
import tensorflow as tf
import h5py

def main():
	sess = tf.Session()
	# inputs
	U=tf.linspace(1.0,10.0,500)
	U =tf.reshape(U,[2,5,5,5,2])
	#encoder initial
	X_controlP_number = 4
	Y_controlP_number = 4
	Z_controlP_number = 4
	tps_out_size = (40,40,40)
	#decoder initial
	X_controlP_number_D = 4
	Y_controlP_number_D = 4
	Z_controlP_number_D = 4
	out_size_D = (40, 40,40)
	# encoder
	transform = transformer(U,U,X_controlP_number,Y_controlP_number,Z_controlP_number,tps_out_size)
	conv1,T,cp= transform.TPS_transformer(U,U)
	#decoder
	inverse_trans = inverse_transformer(conv1,X_controlP_number_D,Y_controlP_number_D,Z_controlP_number_D,out_size_D)
	conv2 = inverse_trans.TPS_decoder(conv1,conv1,T)
if __name__ == "__main__":
    main()

