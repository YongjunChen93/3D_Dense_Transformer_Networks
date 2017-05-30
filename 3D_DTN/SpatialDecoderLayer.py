import tensorflow as tf
import time
import datetime
sess = tf.Session()
class inverse_transformer(object):
	def __init__(self,U,Column_controlP_number_D,Row_controlP_number_D,Z_controlP_number_D, out_size):
		self.num_batch = U.shape[0].value
		self.depth = U.shape[1].value
		self.height = U.shape[2].value
		self.width = U.shape[3].value
		self.num_channels = U.shape[4].value

		self.out_height = self.height
		self.out_width = self.width
		self.out_depth = self.depth
		self.out_size = out_size

		self.X_controlP_number_D = Column_controlP_number_D
		self.Y_controlP_number_D = Row_controlP_number_D
		self.Z_controlP_number_D = Z_controlP_number_D

	def _repeat(self,x, n_repeats, type):
		with tf.variable_scope('_repeat'):
			rep = tf.transpose(
			    tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
			rep = tf.cast(rep, type)
			x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
			return tf.reshape(x, [-1])

	def _bilinear_interpolate(self,im, im_org, x, y,z):
		with tf.variable_scope('_interpolate'):
            # constants
			x = tf.cast(x, 'float32')
			y = tf.cast(y, 'float32')
			z = tf.cast(z, 'float32')
			height_f = tf.cast(self.height, 'float32')
			width_f = tf.cast(self.width, 'float32')
			depth_f = tf.cast(self.depth, 'float32')
			zero = tf.zeros([], dtype='int32')
			max_z = tf.cast(tf.shape(im)[1] - 1, 'int32')
			max_y = tf.cast(tf.shape(im)[2] - 1, 'int32')
			max_x = tf.cast(tf.shape(im)[3] - 1, 'int32')
			# scale indices from [-1, 1] to [0, width/height]
			# scale indices from [-1, 1] to [0, width/height]
			x = (x + 1.0)*(width_f) / 2.0
			y = (y + 1.0)*(height_f) / 2.0
			z = (z + 1.0)*(depth_f) / 2.0
			# do sampling
			x0 = tf.cast(tf.floor(x), 'int32')
			x1 = x0 + 1
			y0 = tf.cast(tf.floor(y), 'int32')
			y1 = y0 + 1
			z0 = tf.cast(tf.floor(z), 'int32')
			z1 = z0 + 1
			x0 = tf.clip_by_value(x0, zero, max_x)
			x1 = tf.clip_by_value(x1, zero, max_x)
			y0 = tf.clip_by_value(y0, zero, max_y)
			y1 = tf.clip_by_value(y1, zero, max_y)
			z0 = tf.clip_by_value(z0, zero, max_z)
			z1 = tf.clip_by_value(z1, zero, max_z)
			dim2 = self.width
			dim1 = self.width*self.height
			dim0 = self.width*self.height*self.depth
			base = self._repeat(tf.range(self.num_batch)*dim0, self.out_height*self.out_width*self.out_depth,'int32')
			base_z0 = base + z0*dim1
			base_z1 = base + z1*dim1
			base_y0 = y0*dim2
			base_y1 = y1*dim2
			#lower layer
			idx_a = tf.expand_dims(base_z0 + base_y0 + x0,1)
			idx_b = tf.expand_dims(base_z0 + base_y1 + x0,1)
			idx_c = tf.expand_dims(base_z0 + base_y0 + x1,1)
			idx_d = tf.expand_dims(base_z0 + base_y1 + x1,1)
			#upper layer
			idx_e = tf.expand_dims(base_z1 + base_y0 + x0,1)
			idx_f = tf.expand_dims(base_z1 + base_y1 + x0,1)
			idx_g = tf.expand_dims(base_z1 + base_y0 + x1,1)
			idx_h = tf.expand_dims(base_z1 + base_y1 + x1,1)
			# use indices to lookup pixels in the flat image and restore
			# channels dim
			im_flat = tf.reshape(im, tf.stack([-1, self.num_channels]))
			im_flat = tf.cast(im_flat, 'float32')
			Ia = tf.scatter_nd(idx_a, im_flat, [self.num_batch*self.out_height*self.out_width*self.out_depth, self.num_channels])
			Ib = tf.scatter_nd(idx_b, im_flat, [self.num_batch*self.out_height*self.out_width*self.out_depth, self.num_channels])
			Ic = tf.scatter_nd(idx_c, im_flat, [self.num_batch*self.out_height*self.out_width*self.out_depth, self.num_channels])
			Id = tf.scatter_nd(idx_d, im_flat, [self.num_batch*self.out_height*self.out_width*self.out_depth, self.num_channels])
			Ie = tf.scatter_nd(idx_e, im_flat, [self.num_batch*self.out_height*self.out_width*self.out_depth, self.num_channels])
			If = tf.scatter_nd(idx_f, im_flat, [self.num_batch*self.out_height*self.out_width*self.out_depth, self.num_channels])
			Ig = tf.scatter_nd(idx_g, im_flat, [self.num_batch*self.out_height*self.out_width*self.out_depth, self.num_channels])
			Ih = tf.scatter_nd(idx_h, im_flat, [self.num_batch*self.out_height*self.out_width*self.out_depth, self.num_channels])			

			x0_f = tf.cast(x0, 'float32')
			x1_f = tf.cast(x1, 'float32')
			y0_f = tf.cast(y0, 'float32')
			y1_f = tf.cast(y1, 'float32')
			z0_f = tf.cast(z0, 'float32')
			z1_f = tf.cast(z1, 'float32')

			wa = tf.scatter_nd(idx_a, tf.expand_dims(((x1_f-x) * (y1_f-y) * (z1_f-z)), 1), [self.num_batch*self.out_height*self.out_width*self.out_depth, 1])
			wb = tf.scatter_nd(idx_b, tf.expand_dims(((x1_f-x) * (y-y0_f) * (z1_f-z)), 1), [self.num_batch*self.out_height*self.out_width*self.out_depth, 1])
			wc = tf.scatter_nd(idx_c, tf.expand_dims(((x-x0_f) * (y1_f-y) * (z1_f-z)), 1), [self.num_batch*self.out_height*self.out_width*self.out_depth, 1])
			wd = tf.scatter_nd(idx_d, tf.expand_dims(((x-x0_f) * (y-y0_f) * (z1_f-z)), 1), [self.num_batch*self.out_height*self.out_width*self.out_depth, 1])
			we = tf.scatter_nd(idx_e, tf.expand_dims(((x1_f-x) * (y1_f-y) * (z-z0_f)), 1), [self.num_batch*self.out_height*self.out_width*self.out_depth, 1])
			wf = tf.scatter_nd(idx_f, tf.expand_dims(((x1_f-x) * (y-y0_f) * (z-z0_f)), 1), [self.num_batch*self.out_height*self.out_width*self.out_depth, 1])
			wg = tf.scatter_nd(idx_g, tf.expand_dims(((x-x0_f) * (y1_f-y) * (z-z0_f)), 1), [self.num_batch*self.out_height*self.out_width*self.out_depth, 1])
			wh = tf.scatter_nd(idx_h, tf.expand_dims(((x-x0_f) * (y-y0_f) * (z-z0_f)), 1), [self.num_batch*self.out_height*self.out_width*self.out_depth, 1])

			value_all = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id, we*Ie, wf*If, wg*Ig, wh*Ih])
			weight_all = tf.clip_by_value(tf.add_n([wa, wb, wc, wd,we, wf, wg, wh]),1e-5,1e+10)
			flag = tf.less_equal(weight_all, 1e-5* tf.ones_like(weight_all))
			flag = tf.cast(flag, tf.float32)
			im_org = tf.reshape(im_org, [-1,self.num_channels])
			output = tf.add(tf.div(value_all, weight_all), tf.multiply(im_org, flag))
			return output

	def _meshgrid(self):
		with tf.variable_scope('_meshgrid'):
			sess = tf.Session()
			x_use = tf.linspace(-1.0, 1.0, self.out_width)
			y_use = tf.linspace(-1.0, 1.0, self.out_height)
			z_use = tf.linspace(-1.0, 1.0, self.out_depth)
			x_t = tf.tile(x_use,[self.out_width*self.out_depth])
			y_t = tf.tile(self._repeat(y_use,self.out_height,'float32'),[self.out_depth])
			z_t = self._repeat(z_use,self.out_height*self.out_width,'float32')

			x_t_flat = tf.reshape(x_t, (1, -1))
			y_t_flat = tf.reshape(y_t, (1, -1))
			z_t_flat = tf.reshape(z_t, (1, -1))
			px,py,pz = tf.stack([x_t_flat],axis=2),tf.stack([y_t_flat],axis=2),tf.stack([z_t_flat],axis=2)
			#source control points
			x,y,z = tf.linspace(-1.,1.,self.X_controlP_number_D),tf.linspace(-1.,1.,self.Y_controlP_number_D),tf.linspace(-1.,1.,self.Z_controlP_number_D)
			x   = tf.tile(x,[self.Y_controlP_number_D*self.Z_controlP_number_D])
			y   = tf.tile(self._repeat(y,self.X_controlP_number_D,'float32'),[self.Z_controlP_number_D])
			z   = self._repeat(z,self.X_controlP_number_D*self.Y_controlP_number_D,'float32')
			xs,ys,zs = tf.transpose(tf.reshape(x,(-1,1))),tf.transpose(tf.reshape(y,(-1,1))),tf.transpose(tf.reshape(z,(-1,1)))
			cpx,cpy,cpz = tf.transpose(tf.stack([xs],axis=2),perm=[1,0,2]),tf.transpose(tf.stack([ys],axis=2),perm=[1,0,2]),tf.transpose(tf.stack([zs],axis=2),perm=[1,0,2])
			px, cpx = tf.meshgrid(px,cpx);py, cpy = tf.meshgrid(py,cpy); pz, cpz = tf.meshgrid(pz,cpz)        
			#Compute distance R
			Rx,Ry,Rz = tf.square(tf.subtract(px,cpx)),tf.square(tf.subtract(py,cpy)),tf.square(tf.subtract(pz,cpz))
			
			R = tf.add(tf.add(Rx,Ry),Rz)        
			R = tf.multiply(R,tf.log(tf.clip_by_value(R,1e-10,1e+10)))
			#Source coordinates
			ones = tf.ones_like(x_t_flat) 
			grid = tf.concat([ones, x_t_flat, y_t_flat,z_t_flat,R],0)
			return grid

	def _transform(self, T, U, U_org):
		with tf.variable_scope('_transform'):
			T = tf.reshape(T, (-1, 3, self.X_controlP_number_D*self.Y_controlP_number_D*self.Z_controlP_number_D+4))
			T = tf.cast(T, 'float32')
			# grid of (x_t, y_t, 1), eq (1) in ref [1]
			# 19 * (H * W )
			grid = self._meshgrid()
			grid = tf.expand_dims(grid, 0)
			grid = tf.reshape(grid, [-1])
			grid = tf.tile(grid, tf.stack([self.num_batch]))
			grid = tf.reshape(grid, tf.stack([self.num_batch, self.X_controlP_number_D*self.Y_controlP_number_D*self.Z_controlP_number_D+4, -1]))
			T_g = tf.matmul(T, grid)
			x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
			y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
			z_s = tf.slice(T_g, [0, 2, 0], [-1, 1, -1])
			#print('x_s', sess.run(x_s))
			x_s_flat = tf.reshape(x_s, [-1])
			y_s_flat = tf.reshape(y_s, [-1])
			z_s_flat = tf.reshape(z_s, [-1])
			output_transformed = self._bilinear_interpolate(U,U_org,x_s_flat,y_s_flat,z_s_flat)
			output = tf.reshape(output_transformed, tf.stack([self.num_batch, self.out_depth,self.out_height, self.out_width,self.num_channels]))
			return output
			
	def TPS_decoder(self,U, U_org, T,name='SpatialDecoderLayer', **kwargs):
		with tf.variable_scope(name):
			output = self._transform(T, U, U_org)
			return output





