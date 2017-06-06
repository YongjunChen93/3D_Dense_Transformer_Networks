# 3D Dense Transformer Networks

This is a extended work of our recent work, "Dense Transformer Networks". Please check the [paper](https://arxiv.org/abs/1705.08881) for details.

In this work we extended the Dense Transformer Networks from 2 dimension to 3 dimension. The third dimension could be either space or time.

## What is 3D Dense Transformer Networks

### Affine transformation function demo

![images](https://github.com/JohnYC1995/3D_Dense_Transformer_Networks/blob/master/images/Affine_demo.png)

### TPS transformation function demo

![images](https://github.com/JohnYC1995/3D_Dense_Transformer_Networks/blob/master/images/TPS_demo.png)

### Data Prepare

In the experiments, we first set the original dataset into a certain size and save as hdf5 format. Then we do the data augmentation on the 3D image dataset to enrich the dataset. 
#### Show 3D data
To Show the 3D Data, one just need to use show_data() function in Data_prepare file.
#### Save data into current format
To Save the data into corresponding format, one just need to use save() function in Data_prepare file.
A example work is on the pre_data_main file. 
#### Data augmentation
The augmentation type curently support:
```
* Flip up and down on X, Y, Z axises
* Flip left and right on X, Y, Z axises
* rotation 90,180,270 degrees on X, Y axises.
```
So for one input data, it will be at most enlargeed into 64 datas.  This part code is in data_augmentation file.

### TPS_transformer

```
Parameters  

* U: the input of spatial transformer.  
* U_local: the input of localization networks.  
```

### TPS_decoder

```
Parameters  

* U: the input of spatial deocder transformer.  
* U_org: the original feature maps to fill the missing pixels.  
* T: the transformation shared with TPS_transformer.  
```
### Simple Running Example
```
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

```
