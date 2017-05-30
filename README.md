# 3D Dense Transformer Networks

This is a extended work of our recent work, "Dense Transformer Networks". Please check the [paper](https://arxiv.org/abs/1705.08881) for details.

## Experimental results:
We perform our experiment on two datasets to compare the baseline U-Net model and the proposed DTN model.

## How to use

![image](https://github.com/divelab/dtn/blob/master/results/architecture.PNG)

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

