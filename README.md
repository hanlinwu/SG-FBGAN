# Remote Sensing Image Super-Resolution via Visual Attention Mechanism and Feedback GANs

by Hanlin Wu, Libao Zhang, and Jie Ma, details are in paper.

## Introduction
This repository is build for the proposed VA-FBGAN, which contains full training and testing code. 

## Usage

### Requirement:

Python = 3.6

TensorFlow = 1.14.0

TensorLayer = 1.11.0

opency-python

wget

numpy

easydict

tqdm

### Clone the repository:

> git clone ...

### Train
Download and prepare the ImageNet dataset (ILSVRC2012) and symlink the path to it as follows (you can alternatively modify the relevant path specified in folder config):

> python train.py --opt config/va_fbgan_x3_3it.json

### Test
Download trained VA-FBGAN models and put them under folder specified in config or modify the specified paths, and then do testing:

> python predict.py --opt exp_name

### Performance
coming soon...

## Citation
coming soon...

## To do list

1. 修改文章 x8 的channel 数为64, C = 3, D = 4。
2. 修改curriculum learning 的gan loss 设置。