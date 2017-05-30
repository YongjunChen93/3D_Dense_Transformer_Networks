# 3D Dense Transformer Networks

This is a extended work of our recent work, "Dense Transformer Networks". Please check the [paper](https://arxiv.org/abs/1705.08881) for details.

In this work we extended the Dense Transformer Networks from 2 dimension to 3 dimension. The third dimension could be either space or time.

## What is 3D Dense Transformer Networks
A framework of our 3D DTN.

![images](https://github.com/JohnYC1995/3D_Dense_Transformer_Networks/blob/master/images/3D_DTN_framework.png)

### Affine transformation function demo

![images](https://github.com/JohnYC1995/3D_Dense_Transformer_Networks/blob/master/images/Affine_demo.png)

### TPS transformation function demo

![images](https://github.com/JohnYC1995/3D_Dense_Transformer_Networks/blob/master/images/TPS_demo.png)

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
