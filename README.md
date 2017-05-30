# 3D Dense Transformer Networks

This is a extended work of our recent work, "Dense Transformer Networks". Please check the [paper](https://arxiv.org/abs/1705.08881) for details.

## What is 3D Dense Transformer Networks
A framework of our 3D DTN.

![pdf](https://github.com/JohnYC1995/3D_Dense_Transformer_Networks/blob/master/images/3D_DTN.pdf)


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

### TPS transformation function demo

![pdf](https://github.com/JohnYC1995/3D_Dense_Transformer_Networks/blob/master/images/TPS_sample.pdf)

