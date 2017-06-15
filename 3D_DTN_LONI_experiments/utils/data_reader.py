import glob
import h5py
import random
import tensorflow as tf
import numpy as np
from .img_utils import get_images


"""
This module provides three data reader: directly from file, from h5 database, use channel

h5 database is recommended since it could enable very data feeding speed
"""


class FileDataReader(object):

    def __init__(self, data_dir, input_height, input_width, height, width,
                 batch_size):
        self.data_dir = data_dir
        self.input_height, self.input_width = input_height, input_width
        self.height, self.width = height, width
        self.batch_size = batch_size
        self.image_files = glob.glob(data_dir+'*')
		

    def next_batch(self, batch_size):
        sample_files = np.random.choice(self.image_files, batch_size)
        images = get_images(
            sample_files, self.input_height, self.input_width,
            self.height, self.width)
        return images


class H5DataLoader(object):

    def __init__(self, data_path, is_train=True):
        self.is_train = is_train
        data_file = h5py.File(data_path, 'r')
        self.images, self.labels = data_file['X'], data_file['Y']
        self.gen_indexes()

    def gen_indexes(self):
        if self.is_train:
            self.indexes = np.random.permutation(range(self.images.shape[0]))
        else:
            self.indexes = np.array(range(self.images.shape[0]))
        self.cur_index = 0

    def next_batch(self, batch_size):
        next_index = self.cur_index+batch_size
        cur_indexes = list(self.indexes[self.cur_index:next_index])
        self.cur_index = next_index
        if len(cur_indexes) < batch_size and self.is_train:
            self.gen_indexes()
            self.cur_index = batch_size-len(cur_indexes)
            cur_indexes += list(self.indexes[:batch_size-len(cur_indexes)])
        cur_indexes = sorted(set(cur_indexes))
        try:
            return self.images[cur_indexes], self.labels[cur_indexes]
        except Exception as e:
            print(e)
            return np.empty(0), np.empty(0)


class H53DDataLoader(object):

    def __init__(self, data_path, shape, is_train=True):
        self.is_train = is_train
        self.d, self.h, self.w = shape[1:-1] 
        self.dataset_name = []
        for line in open(data_path):
            name = line.split('\n',1)
            self.dataset_name.append(name[0])
        self.index = 0
        self.sub_batch_index = 0
    def next_batch(self, batch_size,gap=None):
        data_path = self.dataset_name[self.index]
        print("data_path===================",data_path)
        data_file = h5py.File(data_path, 'r')
        self.images, self.labels = data_file['data'][self.sub_batch_index,:,:,:], data_file['label'][self.sub_batch_index,:,:,:]
        self.sub_batch_index +=1
        if self.sub_batch_index >=  data_file['data'].shape[0]:
            self.sub_batch_index = 0
            self.index += 1
        image_batches = []
        label_batches = []
        if self.is_train == True:
            print("train=====")
            if self.index >= len(self.dataset_name):
                self.index = 0
            batches_ids = set()
            self.t_d, self.t_h, self.t_w = self.images.shape
            while len(batches_ids) < batch_size:
                d = random.randint(0, self.t_d-self.d)
                h = random.randint(0, self.t_h-self.h)
                w = random.randint(0, self.t_w-self.w)
                batches_ids.add(( d, h, w))
            for d, h, w in batches_ids:
                image_batches.append(
                    self.images[d:d+self.d, h:h+self.h, w:w+self.w])
                label_batches.append(
                    self.labels[d:d+self.d, h:h+self.h, w:w+self.w])
            images = np.expand_dims(np.stack(image_batches, axis=0), axis=-1)
#            images = np.transpose(images, (0, 3, 1, 2, 4))
            labels = np.stack(label_batches, axis=0)
#            labels = np.transpose(labels, (0, 3, 1, 2))
            return images, labels

    def generate_data(self,index,batch_size,patch_shape,gap=None):
        self.index = index
        data_path = self.dataset_name[self.index]
        data_file = h5py.File(data_path, 'r')
        self.data_all, self.label_all = np.transpose(np.array(data_file['data']),(2,0,1)),  np.transpose(np.array(data_file['label']),(2,0,1))
        self.batch_size =batch_size
        self.patch_shape = patch_shape

        image_batches = []
        label_batches = []
        s_patch = self.batch_size[0] 
        h_patch = self.batch_size[1]
        w_patch = self.batch_size[2]

        d_gap = self.patch_shape[0]
        h_gap = self.patch_shape[1]
        w_gap = self.patch_shape[2]

        s_input = self.data_all.shape[0]
        h_input = self.data_all.shape[1]
        w_input = self.data_all.shape[2]

        if gap == None:
            d_gap = d_gap
            w_gap = w_gap
            h_gap = h_gap
        else:
            d_gap = gap[0]
            w_gap = gap[1]
            h_gap = gap[2]

        for i in range(1,2):
            for s_start in range(0,s_input-d_gap+1,d_gap):
                s_start =min(s_start,s_input-s_patch)
                s_end = s_start+s_patch
                for h_start in range(0,h_input-h_gap+1,h_gap):
                    h_start = min(h_start,h_input-h_patch)
                    h_end = h_start+h_patch
                    for w_start in range(0,w_input-w_gap+1,w_gap):
                        #print("w_start",w_start)
                        w_start = min(w_start,w_input-w_patch)
                        w_end = w_start+w_patch
                        X = self.data_all[s_start:s_end,h_start:h_end,w_start:w_end]
                        Y = self.label_all[s_start:s_end,h_start:h_end,w_start:w_end]
                        images = np.expand_dims(np.expand_dims(np.stack(X, axis=0), axis=-1),axis=0)
                        labels = np.expand_dims(np.stack(Y, axis=0),axis=0)
                        yield images,labels


class QueueDataReader(object):

    def __init__(self, sess, data_dir, data_list, input_size, class_num,
                 name, data_format):
        self.sess = sess
        self.scope = name + '/data_reader'
        self.class_num = class_num
        self.channel_axis = 3
        images, labels = self.read_data(data_dir, data_list)
        images = tf.convert_to_tensor(images, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.string)
        queue = tf.train.slice_input_producer(
            [images, labels], shuffle=True, name=self.scope+'/slice')
        self.image, self.label = self.read_dataset(
            queue, input_size, data_format)

    def next_batch(self, batch_size):
        image_batch, label_batch = tf.train.shuffle_batch(
            [self.image, self.label], batch_size=batch_size,
            num_threads=4, capacity=50000, min_after_dequeue=10000,
            name=self.scope+'/batch')
        return image_batch, label_batch

    def read_dataset(self, queue, input_size, data_format):
        image = tf.image.decode_jpeg(
            tf.read_file(queue[0]), channels=3, name=self.scope+'/image')
        label = tf.image.decode_png(
            tf.read_file(queue[1]), channels=1, name=self.scope+'/label')
        image = tf.image.resize_images(image, input_size)
        label = tf.image.resize_images(label, input_size, 1)
        if data_format == 'NCHW':
            self.channel_axis = 1
            image = tf.transpose(image, [2, 0, 1])
            label = tf.transpose(label, [2, 0, 1])
        image -= tf.reduce_mean(tf.cast(image, dtype=tf.float32),
                                (0, 1), name=self.scope+'/mean')
        return image, label

    def read_data(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            images, labels = [], []
            for line in f:
                image, label = line.strip('\n').split(' ')
                images.append(data_dir + image)
                labels.append(data_dir + label)
        return images, labels

    def start(self):
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(
            coord=self.coord, sess=self.sess)

    def close(self):
        self.coord.request_stop()
        self.coord.join(self.threads)

