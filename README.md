# Remote Sensing Image Super-Resolution via Visual Attention Mechanism and Feedback GANs

by Hanlin Wu, Libao Zhang, and Jie Ma, details are in paper.

## Introduction
This repository is build for the proposed VA-FBGAN, which contains full training and testing code. 

![framework](/_images/framework.png)

## Performance
coming soon...

## Usage

### Clone the repository:

> git clone https://github.com/BNUAI/VA-FBGAN.git
 
### Requirement:

> pip install -r requirements.txt

### Test with our pre-trained models:

Download the pre-trained VA-FBGAN models, and then do testing:

> python predict.py --opt exp_name

### Train with our GeoEye dataset:

> python train.py --opt config/va_fbgan_x3_BI.json

### Train with your own dataset:

> python train.py --opt config/va_fbgan_x3_BI.json

## Citation
coming soon...