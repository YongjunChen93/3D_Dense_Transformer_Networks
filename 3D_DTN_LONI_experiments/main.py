import os
import time
import argparse
import tensorflow as tf
from network import DenseTransformerNetwork

"""
This file provides configuration to build U-NET for semantic segmentation.

"""

def configure():
    flags = tf.app.flags
    # data
    flags.DEFINE_string('data_dir', '/tempspace/ychen7/Research3/augment_data/', 'Name of data directory')
    flags.DEFINE_string('train_data', 'LONI_train/train.txt', 'Training data')
    flags.DEFINE_string('valid_data', 'LONI_valid/valid.txt', 'Validation data')
    flags.DEFINE_string('test_data', 'testing3d.h5', 'Testing data')
    flags.DEFINE_string('data_type', '3D', '2D data or 3D data')
    # training
    flags.DEFINE_integer('max_step', 500000, '# of step for training')
    flags.DEFINE_integer('test_interval', 100, '# of interval to test a model')
    flags.DEFINE_integer('save_interval', 1000, '# of interval to save a model')
    flags.DEFINE_integer('summary_interval', 100, '# of step to save the summary')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_integer('batch', 1, 'batch size')
    flags.DEFINE_integer('channel', 1, 'channel size')
    flags.DEFINE_integer('depth', 32, 'depth size')
    flags.DEFINE_integer('height', 128, 'height size')
    flags.DEFINE_integer('width', 128, 'width size')
    #validing
    flags.DEFINE_integer('valid_start_epoch',1000,'start step to test a model')
    flags.DEFINE_string('valid_end_epoch',100000,'end step to test a model')
    flags.DEFINE_string('valid_stride_of_epoch',1000,'stride to test a model')
    # predict
    flags.DEFINE_integer('d_gap',30,'predict depth gap')
    flags.DEFINE_integer('w_gap',100,'predict width gap')
    flags.DEFINE_integer('h_gap',100,'predict height gap')
    flags.DEFINE_integer('data_height',142,'predict data_height')
    flags.DEFINE_integer('data_width',149,'predict data_width')
    flags.DEFINE_integer('predict_batch',181,'predict batch')
    flags.DEFINE_integer('test_step', 84000, 'Test or predict model at this step')
    # Debug
    flags.DEFINE_string('logdir', './logdir', 'Log dir')
    flags.DEFINE_string('modeldir', './modeldir', 'Model dir')
    flags.DEFINE_string('sampledir', './samples/', 'Sample directory')
    flags.DEFINE_string('predict_name','prediction_result.h5','prediction result name')
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')
    flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')
    # network architecture
    flags.DEFINE_integer('network_depth', 5, 'network depth for U-Net')
    flags.DEFINE_integer('class_num', 57, 'output class number')
    flags.DEFINE_integer('start_channel_num', 32,
                         'start number of outputs for the first conv layer')
    flags.DEFINE_string(
        'conv_name', 'conv',
        'Use which conv op in decoder: conv or ipixel_cl')
    flags.DEFINE_string(
        'deconv_name', 'deconv',
        'Use which deconv op in decoder: deconv, pixel_dcl, ipixel_dcl')
    # Dense Transformer Networks
    flags.DEFINE_boolean('add_dtn', True,
        'add Dense Transformer Networks or not')
    flags.DEFINE_integer('dtn_location', 3,'The Dense Transformer Networks location')
    flags.DEFINE_string('control_points_ratio', 2,
        'Setup the ratio of control_points comparing with the Dense transformer networks input size')    
    # fix bug of flags
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS

def average_list(inputlist):
    sum_num = reduce(add,inputlist)
    return sum_num/len(inputlist)

def valid(model,datasize,batchsize):
    conf =  configure()
    epoch_accuracys = []
    epoch_loss      = []
    for epoch in range(conf.valid_start_epoch,conf.valid_end_epoch,conf.valid_stride_of_epoch):
        average_accuracy = []
        average_loss     = []
        for data_index in range(0,datasize):
            for sub_batch_index in range(0,batchsize):
                print("predict epoch:",epoch,"dataset index:",data_index,\
                      "sub_batch_index",sub_batch_index)
                accuracy,loss = model.predict(epoch, data_index,sub_batch_index)
                average_accuracy.append(accuracy)
                average_loss.append(loss)
                print("accuracy",average_accuracy)
                print("loss",average_loss)
        average_accuracy = average_list(average_list)
        average_loss = average_list(average_loss)
        print("average_accuracy----->",average_accuracy)
        print("average_loss------>",average_loss)
        epoch_accuracys.append((epoch,average_accuracy))
        epoch_loss.append((epoch,average_loss))
    print("accuracy on each epoch----->", epoch_accuracys)
    print("loss on each epoch------>", epoch_loss)

def predict(model,datasize,batchsize):
    average_accuracy = []
    average_loss     = []
    for data_index in range(0,datasize):
        for sub_batch_index in range(0,batchsize):
            print("predict dataset index:",data_index,"sub_batch_index",sub_batch_index)
            accuracy,loss = model.predict(configure().test_step, data_index,sub_batch_index)
            average_accuracy.append(accuracy)
            average_loss.append(loss)
            print("accuracy",average_accuracy)
            print("loss",average_loss)
    average_accuracy = average_list(average_list)
    average_loss = average_list(average_loss)
    print("average_accuracy----->",average_accuracy)
    print("average_loss------>",average_loss)

def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', dest='action', type=str, default='train',
                        help='actions: train, test, or predict')
    args = parser.parse_args()
    if args.action not in ['train', 'test', 'predict']:
        print('invalid action: ', args.action)
        print("Please input a action: train, test, or predict")
    elif args.action == 'test':
        model = DenseTransformerNetwork(tf.Session(), configure())
        valid(model,16,10)
    elif args.action == 'predict':
        model = DenseTransformerNetwork(tf.Session(), configure())
        predict(model,16,10)
    else:
        model = DenseTransformerNetwork(tf.Session(), configure())
        getattr(model, args.action)()


if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    tf.app.run()
