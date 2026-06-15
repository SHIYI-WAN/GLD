# GLD： Gabor Convolutional Network for 2D Line Descriptors
This repository contains the pre-trained network implementation of the following paper:
# Getting started:
Conda environment
```
conda create -n GLD python=3.8.20
conda activate GLD
pip install -r requirements.txt
```
Our GLD network needs RGB line patches of 192x128(HxW) pixels as input (an example is provided inside `IN_OUT_DATA`)

To obtain the descriptors, first you need to use an arbitrary line detector to get detected lines, and then you can extract the line patch using a MATLAB function with their endpoints:
```
imCropFromMidpoint.m
```
Then run the inference code:
```
Inference_patch_descriptor.py
```
#Pretrained model
We provide the pre-trained model in the `model/checkp/checkpoint_GLD.pth.tar', which is the GLD we propose in the paper
