# 3D Dense Transformer Networks

This is a extended work of our recent work, "Dense Transformer Networks". Please check the [paper](https://arxiv.org/abs/1705.08881) for details.

In this work we extended the Dense Transformer Networks from 2 dimension to 3 dimension. The third dimension could be either space or time.

## What is 3D Dense Transformer Networks

### Affine transformation function demo

![images](https://github.com/JohnYC1995/3D_Dense_Transformer_Networks/blob/master/images/Affine_demo.png)

### TPS transformation function demo

![images](https://github.com/JohnYC1995/3D_Dense_Transformer_Networks/blob/master/images/TPS_demo.png)

### Data Preparation

In the experiments, I first set the original 3D dataset into a certain size and save as hdf5 format. Then do the data augmentation on the 3D image dataset to enrich the dataset. 
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

### Dense Transformer Networks Simple Running Example

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
