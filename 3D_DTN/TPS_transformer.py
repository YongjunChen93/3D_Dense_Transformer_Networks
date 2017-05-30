import tensorflow as tf
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
from ops import *
sess = tf.Session()
class transformer(object):
    def __init__(self,U,U_local,X_controlP_number,Y_controlP_number,Z_controlP_number,out_size):
        self.initial = np.array([[-5., -0.4, 0.4, 5., -5., -0.4, 0.4, 5., -5., -0.4, 0.4, 5., -5., -0.4, 0.4, 5.],[-5., -5., -5., -5., -0.4, -0.4, -0.4, -0.4, 0.4, 0.4, 0.4, 0.4, 5., 5., 5.,5.]])
        self.local_num_batch = U_local.shape[0].value
        self.local_depth = U_local.shape[1].value
        self.local_height = U_local.shape[2].value
        self.local_width = U_local.shape[3].value
        self.local_num_channels = U_local.shape[4].value

        self.num_batch = U.shape[0].value
        self.depth = U.shape[1].value
        self.height = U.shape[2].value
        self.width = U.shape[3].value
        self.num_channels = U.shape[4].value
        
        self.out_height = self.height
        self.out_width = self.width
        self.out_depth = self.depth
        self.out_size = out_size
        self.X_controlP_number = X_controlP_number
        self.Y_controlP_number = Y_controlP_number
        self.Z_controlP_number = Z_controlP_number


    def _local_Networks(self,input_dim,x):
        with tf.variable_scope('_local_Networks'):
            x = tf.reshape(x,[-1,self.local_height*self.local_width*self.local_depth*self.local_num_channels])
            W_fc_loc1 = weight_variable([self.local_height*self.local_width*self.local_depth*self.local_num_channels, 20])
            b_fc_loc1 = bias_variable([20])
            W_fc_loc2 = weight_variable([20, 32])
            initial = self.initial.astype('float32')
            initial = initial.flatten()
            b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')
            h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1)
            h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1, W_fc_loc2) + b_fc_loc2)
            #temp use
            x = [-1,-0.3,0.3,1]
            y = [-1,-0.3,0.3,1]
            z = [-1,-0.3,0.3,1]
            x_s = tf.tile(x,[self.Y_controlP_number*self.Z_controlP_number])
            y_s = tf.tile(self._repeat(y,self.X_controlP_number,'float32'),[self.Z_controlP_number])
            z_s = self._repeat(z,self.X_controlP_number*self.Y_controlP_number,'float32')
            h_fc_loc2 = tf.concat([x_s,y_s,z_s],0)
            h_fc_loc2 = tf.tile(h_fc_loc2,[self.num_batch])
            h_fc_loc2 = tf.reshape(h_fc_loc2,[self.num_batch,-1])
            #2*(4*4*4)*3->(2,192)
            return h_fc_loc2

    def _makeT(self,cp):
        with tf.variable_scope('_makeT'):
            cp = tf.reshape(cp,(-1,3,self.X_controlP_number*self.Y_controlP_number*self.Z_controlP_number))
            cp = tf.cast(cp,'float32')       
            N_f = tf.shape(cp)[0]         
            #c_s
            x,y,z = tf.linspace(-1.,1.,self.X_controlP_number),tf.linspace(-1.,1.,self.Y_controlP_number),tf.linspace(-1.,1.,self.Z_controlP_number)
            x   = tf.tile(x,[self.Y_controlP_number*self.Z_controlP_number])
            y   = tf.tile(self._repeat(y,self.X_controlP_number,'float32'),[self.Z_controlP_number])
            z   = self._repeat(z,self.X_controlP_number*self.Y_controlP_number,'float32')

            xs,ys,zs = tf.transpose(tf.reshape(x,(-1,1))),tf.transpose(tf.reshape(y,(-1,1))),tf.transpose(tf.reshape(z,(-1,1)))
            cp_s = tf.concat([xs,ys,zs],0)
            cp_s_trans = tf.transpose(cp_s)
            #print("cp_s_trans",sess.run(cp_s_trans))
            # (4*4*4)*3 -> 64 * 3
            ##===Compute distance R
            xs_trans,ys_trans,zs_trans = tf.transpose(tf.stack([xs],axis=2),perm=[1,0,2]),tf.transpose(tf.stack([ys],axis=2),perm=[1,0,2]),tf.transpose(tf.stack([zs],axis=2),perm=[1,0,2])        
            #print("xs_trans",sess.run(xs_trans))
            #print("ys_trans",sess.run(ys_trans))
            #print("zs_trans",sess.run(zs_trans))
            xs, xs_trans = tf.meshgrid(xs,xs_trans);ys, ys_trans = tf.meshgrid(ys,ys_trans);zs, zs_trans = tf.meshgrid(zs,zs_trans)
            Rx,Ry, Rz = tf.square(tf.subtract(xs,xs_trans)),tf.square(tf.subtract(ys,ys_trans)),tf.square(tf.subtract(zs,zs_trans))
            #print("Rx",sess.run(Rx))
            #print("Ry",sess.run(Ry))
            #print("Rz",sess.run(Rz))
            R = tf.add_n([Rx,Ry,Rz])
            #print("R",sess.run(R))
            R = tf.multiply(R,tf.log(tf.clip_by_value(R,1e-10,1e+10)))
            #print("R",sess.run(R))
            ones = tf.ones([self.Y_controlP_number*self.X_controlP_number*self.Z_controlP_number,1],tf.float32)
            ones_trans = tf.transpose(ones)
            zeros = tf.zeros([4,4],tf.float32)
            Deltas1 = tf.concat([ones, cp_s_trans, R],1)
            Deltas2 = tf.concat([ones_trans,cp_s],0)
            Deltas2 = tf.concat([zeros,Deltas2],1)          
            Deltas = tf.concat([Deltas1,Deltas2],0)
            #print("Deltas",sess.run(Deltas))
            ##get deltas_inv
            Deltas_inv = tf.matrix_inverse(Deltas)
            Deltas_inv = tf.expand_dims(Deltas_inv,0)
            Deltas_inv = tf.reshape(Deltas_inv,[-1])
            Deltas_inv_f = tf.tile(Deltas_inv,tf.stack([N_f]))
            Deltas_inv_f = tf.reshape(Deltas_inv_f,tf.stack([N_f,self.X_controlP_number*self.Y_controlP_number*self.Z_controlP_number+4, -1]))
            cp_trans =tf.transpose(cp,perm=[0,2,1])
            zeros_f_In = tf.zeros([N_f,4,3],tf.float32)
            cp = tf.concat([cp_trans,zeros_f_In],1)
            #print("cp",sess.run(cp))
            #print("Deltas_inv_f",sess.run(Deltas_inv_f))
            T = tf.transpose(tf.matmul(Deltas_inv_f,cp),[0,2,1])
            #print("T",sess.run(T))
            return T

    def _repeat(self,x, n_repeats,type_input):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, type_input)
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(self,im, x, y, z):
        with tf.variable_scope('_interpolate'):
            # constants
            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            z = tf.cast(z, 'float32')
            height_f = tf.cast(self.height, 'float32')
            width_f = tf.cast(self.width, 'float32')
            depth_f = tf.cast(self.depth, 'float32')
            zero = tf.zeros([], dtype='int32')
            max_x = tf.cast(tf.shape(im)[3] - 1, 'int32')
            max_y = tf.cast(tf.shape(im)[2] - 1, 'int32')
            max_z = tf.cast(tf.shape(im)[1] - 1, 'int32')
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
            idx_a = base_z0 + base_y0 + x0
            idx_b = base_z0 + base_y1 + x0
            idx_c = base_z0 + base_y0 + x1
            idx_d = base_z0 + base_y1 + x1
            #upper layer
            idx_e = base_z1 + base_y0 + x0
            idx_f = base_z1 + base_y1 + x0
            idx_g = base_z1 + base_y0 + x1
            idx_h = base_z1 + base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, self.num_channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)
            Ie = tf.gather(im_flat, idx_e)
            If = tf.gather(im_flat, idx_f)
            Ig = tf.gather(im_flat, idx_g)
            Ih = tf.gather(im_flat, idx_h)
            # Finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            z0_f = tf.cast(z0, 'float32')
            z1_f = tf.cast(z1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y) * (z1_f-z)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f) * (z1_f-z)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y) * (z1_f-z)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f) * (z1_f-z)), 1)
            we = tf.expand_dims(((x1_f-x) * (y1_f-y) * (z-z0_f)), 1)
            wf = tf.expand_dims(((x1_f-x) * (y-y0_f) * (z-z0_f)), 1)
            wg = tf.expand_dims(((x-x0_f) * (y1_f-y) * (z-z0_f)), 1)
            wh = tf.expand_dims(((x-x0_f) * (y-y0_f) * (z-z0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id,we*Ie, wf*If, wg*Ig, wh*Ih])
            return output

    def _meshgrid(self):
        with tf.variable_scope('_meshgrid'):
            x_use = tf.linspace(-1.0, 1.0, self.out_height)
            y_use = tf.linspace(-1.0, 1.0, self.out_width)
            z_use = tf.linspace(-1.0, 1.0, self.out_depth)
            x_t = tf.tile(x_use,[self.out_width*self.out_depth])
            y_t = tf.tile(self._repeat(y_use,self.out_height,'float32'),[self.out_depth])
            z_t = self._repeat(z_use,self.out_height*self.out_width,'float32')

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))
            z_t_flat = tf.reshape(z_t, (1, -1))
            px,py,pz = tf.stack([x_t_flat],axis=2),tf.stack([y_t_flat],axis=2),tf.stack([z_t_flat],axis=2)
            #source control points
            x,y,z = tf.linspace(-1.,1.,self.X_controlP_number),tf.linspace(-1.,1.,self.Y_controlP_number),tf.linspace(-1.,1.,self.Z_controlP_number)
            x   = tf.tile(x,[self.Y_controlP_number*self.Z_controlP_number])
            y   = tf.tile(self._repeat(y,self.X_controlP_number,'float32'),[self.Z_controlP_number])
            z   = self._repeat(z,self.X_controlP_number*self.Y_controlP_number,'float32')
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

    def _transform(self, T, input_dim):
        with tf.variable_scope('_transform'): 
            T = tf.reshape(T, (-1, 3, self.X_controlP_number*self.Y_controlP_number*self.Z_controlP_number+4))
            #print("T",T)
            T = tf.cast(T, 'float32')
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            #output size is the same as input size
            grid = self._meshgrid()
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([self.num_batch]))
            grid = tf.reshape(grid, tf.stack([self.num_batch, self.X_controlP_number*self.Y_controlP_number*self.Z_controlP_number+4, -1]))
            #print("grid",sess.run(grid))
            # Transform A x (x_t, y_t,z_t, 1)^T -> (x_s, y_s, z_s)
            T_g = tf.matmul(T, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            z_s = tf.slice(T_g, [0, 2, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])
            z_s_flat = tf.reshape(z_s, [-1])
            input_transformed = self._interpolate(
                input_dim, x_s_flat, y_s_flat,z_s_flat)
            output = tf.reshape(
                input_transformed, tf.stack([self.num_batch, self.out_depth,self.out_height, self.out_width,self.num_channels]))
            return output

    def TPS_transformer(self, U, U_local,name='SpatialTransformer', **kwargs):
        with tf.variable_scope(name):
            cp = self._local_Networks(U,U_local)
            #print("cp ",sess.run(cp))
#            print("cp",cp)
            T= self._makeT(cp)
            output = self._transform(T, U)
            return output,T,cp
