# 3D Dense Transformer Networks

This is a extended work of our recent work, "Dense Transformer Networks". Please check the [paper](https://arxiv.org/abs/1705.08881) for details.

In this work we extended the Dense Transformer Networks from 2 dimension to 3 dimension. The third dimension could be either space or time.

## What is 3D Dense Transformer Networks

### Coorinates transformation

The following two images illustrate how the coordinates in Dense Transformer Networks translate with different transformation functions(Affine and TPS).

1. Affine transformation function demo

![images](https://github.com/JohnYC1995/3D_Dense_Transformer_Networks/blob/master/images/Affine_demo.png)

2. TPS transformation function demo

![images](https://github.com/JohnYC1995/3D_Dense_Transformer_Networks/blob/master/images/TPS_demo.png)

### Interpolation 

The following image shows how to interpolate the input value to the output after the coordinates been translated. In our work, we use the interpolate policy based on bilinear interpolation.

![images](https://github.com/JohnYC1995/3D_Dense_Transformer_Networks/blob/master/images/3D_DTN_framework.png)

## Add 3D Dense Transformer Networks to your Neural Networks

If you want to add the 3D Dense Transformer Networks to your own Networks. You just need to download the codes in ```3D_DTN_Code``` folder. The ```3D_DTN_tests.py``` file will give a very clearly example to add the DTN to your Networks.

Here is the example how to add it:

```
    from Dense_Transformer_Networks_3D import *
    import numpy as np
    import tensorflow as tf

    # sample inputs (Shape: NDHWC)
    U=tf.linspace(1.0,10.0,2*8*8*8*2)
    U =tf.reshape(U,[2,8,8,8,2])

    # parameters setup in network initial
    dtn_input_shape = [2,8,8,8,2]
    control_points_ratio = 2

    # parameters setup initial DTN class
    transform = DSN_Transformer_3D(dtn_input_shape,control_points_ratio)

    # encoder layer
    conv1= transform.Encoder(U,U)

    # decoder layer
    conv2 = transform.Decoder(conv1,conv1)

```
## Add 3D Dense Transformer Networks to a standard U-NET for semantic segmentation

If you just want to use DTN based on U-NET for 3D semantic segmentation, you just need to download the whole codes in ```3D_DTN_LONI_experiments``` folder. And then change the model's setup based on your task on ``` main.py``` 

Here is the instruction of how to set up the standard U-NET with Dense Transformer Networks:

### System requirement

#### Programming language

Python 3.5+

#### Python Packages

tensorflow (CPU) or tensorflow-gpu (GPU), numpy, h5py, os.

### Data Preparation

In our experiments,  I first set the original 3D dataset into a certain size and save as hdf5 format. Then do the data augmentation on the 3D image dataset to enrich the dataset. 

#### Show 3D data

To Show the 3D Data, one just need to use show_data() function in `prepare_data.py` file at Data_prepare direction.

#### Save data 

To Save the data into corresponding format, one just need to use save() function in `prepare_data.py` file at Data_prepare direction.

A example work is on the `pre_data_main.py` file at Data_prepare direction.

#### Data augmentation

The augmentation type curently support:
```
* Flip up and down on X, Y, Z axises
* Flip left and right on X, Y, Z axises
* rotation 90,180,270 degrees on X, Y axises.
```
So for one input data, it will be at most enlargeed into 64 datas.  This part code is in data_augmentation file at Data_prepare direction.

### Configure the network

All network hyperparameters are configured in main.py.

#### Training

max_epoch: how many iterations or steps to train

test_step: how many steps to perform a mini test or validation

save_step: how many steps to save the model

summary_step: how many steps to save the summary

keep_prob: dropout probability

#### Data

data_dir: data directory

train_data: training data location which writen in txt file

valid_data: validation data location which writen in txt file

test_data: h5 file for testing

batch: batch size

channel: input data channel number

depth, height, width: depth, height and width of input data

d_gap, w_gap, h_gap: training patch size of depth, height and width.

#### Debug

logdir: where to store log

modeldir: where to store saved models

sampledir: where to store predicted samples, please add a / at the end for convinience

model_name: the name prefix of saved models

reload_epoch: where to return training

test_epoch: which step to test or predict

random_seed: random seed for tensorflow

#### Network architecture

network_depth: how deep of the U-Net including the bottom layer

class_num: how many classes. Usually number of classes plus one for background

start_channel_num: the number of channel for the first conv layer


conv_name: use which convolutional layer in decoder. We have conv2d for standard convolutional layer, and ipixel_cl for input pixel convolutional layer proposed in our paper.

deconv_name: use which upsampling layer in decoder. We have deconv for standard deconvolutional layer, ipixel_dcl for input pixel deconvolutional layer, and pixel_dcl for pixel deconvolutional layer proposed in our paper.

#### Dense Transformer Networks

add_dtn: add Dense Transformer Netwroks or not.

dtn_location: The Dense Transformer Networks location.

control_points_ratio: the ratio of control_points comparing with the Dense transformer networks input size.

### Training and Testing

#### Start training

After configure the network, we can start to train. Run
```
python main.py
```
The training of a U-Net for semantic segmentation will start. 
#### Training process visualization

We employ tensorboard to visualize the training process.

```
tensorboard --logdir=logdir/
```

The segmentation results including training and validation accuracies,loss and Dice rate of each classes and the prediction outputs are all available in tensorboard.






