# Remote Sensing Image Super-Resolution via Saliency-Guided Feedback GANs

by Hanlin Wu, Libao Zhang, and Jie Ma, details are in paper.

## Introduction
This repository is build for the proposed SG-FBGAN, which contains full training and testing code. 

![framework](/_images/framework.jpg)

## Usage

### Clone the repository:

```
git clone https://github.com/BNUAI/SG-FBGAN.git
```

### Requirement:

- tensorflow==1.14.0
- tensorlayer==1.11.0
- numpy
- easydict
- opencv-python
- tqdm
- wget

```
pip install -r requirements.txt
```

### Test with our pre-trained models:

1. Download the pre-trained SG-FBGAN models.
   
- BI degradation
  - x2: [BI_x2.zip](https://github.com/BNUAI/SG-FBGAN/releases/download/v1.0/BI_x2.zip)
  - x3: [BI_x3.zip](https://github.com/BNUAI/SG-FBGAN/releases/download/v1.0/BI_x3.zip)
  - x4: [BI_x4.zip](https://github.com/BNUAI/SG-FBGAN/releases/download/v1.0/BI_x4.zip)
  - x8: [BI_x8.zip](https://github.com/BNUAI/SG-FBGAN/releases/download/v1.0/BI_x8.zip)
- DN degradation
  - x2: [DN_x2.zip](https://github.com/BNUAI/SG-FBGAN/releases/download/v1.0/DN_x2.zip)
  - x3: [DN_x3.zip](https://github.com/BNUAI/SG-FBGAN/releases/download/v1.0/DN_x3.zip)
  - x4: [DN_x4.zip](https://github.com/BNUAI/SG-FBGAN/releases/download/v1.0/DN_x4.zip)
  - x8: [DN_x8.zip](https://github.com/BNUAI/SG-FBGAN/releases/download/v1.0/DN_x8.zip)

1. Unzip the the downloaded file, and put the pre-trained model on path : `experiments/exp_name`
2. Do testing: 
    ```
    python predict.py --opt exp_name
    ```
    **Note:** The GeoEye-1 dataset will be downloaded automatically. If the download fails, please download it manually from [here](https://github.com/BNUAI/SG-FBGAN/releases/download/v1.0/sr_geo.npz), and then put the downloaded file on path : `data/sr_geo.npz`.

### Train with our GeoEye dataset:

```
python train.py --opt config/va_fbgan_x3_BI.json
```

### Train with your own dataset:

1. change the `datapath` and `savepath` in `data_loader/make_npz.py`, and then make the `.npz` file:
   
   ```
   python data_loader/make_npz.py
   ```
2. change the `data_path` in `config/your_own_config_file.json`.
3. Do training:
   ```
   python train.py --opt config/your_own_config_file.json
   ```

## Contact
- Hanlin Wu (hanlinwu@mail.bnu.edu.cn)