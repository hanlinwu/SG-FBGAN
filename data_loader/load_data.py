import tensorflow as tf
import numpy as np
import os,re,cv2
from tensorlayer.prepro import imresize, crop
import wget

class Data():
    def __init__(self, config):
        self.config = config
        self.batch_size = config.TRAIN.batch_size
        self.lr_img_size = config.MODEL.lr_image_size
        self.upscale = config.MODEL.upscale
        self.hr_img_size = self.lr_img_size * self.upscale
        self.batch_size = config.TRAIN.batch_size
        self.degradation_model = config.MODEL.degradation_model
        self.datapath = self.config.data_path
        self.download()
        self.load_data(config)

    def download(self):
        if not os.path.exists(self.datapath):
            wget.download(self.config.data_url, out = self.datapath)

    def load_data(self, config):
        data = np.load(self.datapath, allow_pickle=True)
        train_raw = data['train'][()]
        valid_raw = data['valid'][()]

        train_hr = train_raw['hr'] 
        train_sal = train_raw['sal_map']

        valid_hr = valid_raw['hr']
        valid_sal = valid_raw['sal_map']
        
        train_data = tf.data.Dataset.from_tensor_slices((train_hr, train_sal))
        valid_data = tf.data.Dataset.from_tensor_slices((valid_hr, valid_sal))
        
        train_data = train_data.map(lambda hr, sal : tf.py_func(self._proc_train_data, [hr, sal], [tf.float64]*5), num_parallel_calls=6)
        valid_data = valid_data.map(lambda hr, sal : tf.py_func(self._proc_valid_data, [hr, sal], [tf.float64]*5), num_parallel_calls=6)
        
        self.train = train_data.batch(self.batch_size, drop_remainder=True).prefetch(1)
        self.train = tf.compat.v1.data.make_initializable_iterator(self.train)
        self.valid = valid_data.batch(1).prefetch(1)
        self.valid = tf.compat.v1.data.make_initializable_iterator(self.valid)
        
        self.train_length = len(train_raw['names'])
        self.valid_length = len(valid_raw['names'])
        self.batch_num = self.train_length // self.batch_size
        self.valid_names = [_.replace(".png", "") for _ in valid_raw['names']]
        
    def _imresize(self, x, w, h = 0, range = [-1., 1.]):
        x = x.astype(np.float32)
        x = (x - range[0]) * 255. / (range[1] - range[0])
        h = w if h == 0 else h 
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        x = imresize(x, size=[w, h], interp='bicubic', mode=None)
        x = (x / 255.) * (range[1] - range[0]) + range[0]
        return x

    def _add_gauss_blur(self, img):
        kernel_size = tuple(self.config.MODEL.kernel_size)
        sigma = self.config.MODEL.sigma
        img = cv2.GaussianBlur(img, kernel_size, sigma)
        img = np.expand_dims(img, axis = 2)
        return img

    def _add_gauss_noise(self, img, level):
        noise = np.random.normal(0,  level / 255., (img.shape[0], img.shape[1]))
        noise = np.expand_dims(noise, axis = 2)
        img_noise = img + noise
        img_noise = img_noise.clip(-1, 1)
        return img_noise
        
    def _proc_train_data(self, hr, sal):
        hr = hr / 127.5 - 1.
        sal = sal / 255.
        hr_sal_concat = np.concatenate((hr, sal), axis=2)
        hr_sal_concat = crop(hr_sal_concat, self.hr_img_size, self.hr_img_size, True)
        hr_img = np.expand_dims(hr_sal_concat[:,:,0], axis = 2)
        hr_sal = np.expand_dims(hr_sal_concat[:,:,1], axis = 2)
        lr_sal = self._imresize(hr_sal, self.lr_img_size, range=[0, 1])

        if self.degradation_model == "DN":
            # degradation with gauss noise
            hr_img_noise = self._add_gauss_noise(hr_img, self.config.MODEL.gauss_level)
            lr_img = self._imresize(hr_img_noise, self.lr_img_size)
            return hr_img, hr_sal, lr_img, lr_sal, hr_img_noise

        elif self.degradation_model == "BD":
            #Gaussian blur
            hr_uint8 = ((hr + 1.) * 127.5).astype(np.uint8)
            hr_blur = self._add_gauss_blur(hr_uint8)
            hr_blur = hr_blur / 127.5 - 1.
            hr_blur_sal_concat = np.concatenate((hr, hr_blur, sal), axis=2)
            hr_blur_sal_concat = crop(hr_blur_sal_concat, self.hr_img_size, self.hr_img_size, True)
            hr_img = np.expand_dims(hr_blur_sal_concat[:,:,0], axis = 2)
            hr_img_blur = np.expand_dims(hr_blur_sal_concat[:,:,1], axis = 2)
            hr_sal = np.expand_dims(hr_blur_sal_concat[:,:,2], axis = 2)
            lr_img = self._imresize(hr_img_blur, self.lr_img_size)
            lr_sal = self._imresize(hr_sal, self.lr_img_size, range=[0, 1])
            return hr_img, hr_sal, lr_img, lr_sal, hr_img_blur

        elif self.degradation_model == "BI":
            # standart degratdation model
            lr_img = self._imresize(hr_img, self.lr_img_size)
            return hr_img, hr_sal, lr_img, lr_sal, np.zeros([1,1,1])
                
    def _proc_valid_data(self, hr, sal):
        hr = hr / 127.5 - 1.
        sal = sal / 255.
        hr_sal_concat = np.concatenate((hr, sal), axis=2)

        upscale = self.upscale
        hr_image_size = hr_sal_concat.shape[0]
        lr_image_size = hr_image_size // upscale
        sr_image_size = lr_image_size * upscale

        hr_sal_concat = hr_sal_concat[0:sr_image_size, 0:sr_image_size]

        hr_img = np.expand_dims(hr_sal_concat[:,:,0], axis = 2)
        hr_sal = np.expand_dims(hr_sal_concat[:,:,1], axis = 2)
        lr_sal = self._imresize(hr_sal, lr_image_size, range = [0, 1])

        if self.degradation_model == "DN":
            # degradation with gauss noise
            hr_img_noise = self._add_gauss_noise(hr_img, self.config.MODEL.gauss_level)
            lr_img = self._imresize(hr_img_noise, lr_image_size)
            return hr_img, hr_sal, lr_img, lr_sal, hr_img_noise
        elif self.degradation_model == "BD":
            #Gaussian blur
            hr_uint8 = ((hr + 1.) * 127.5).astype(np.uint8)
            hr_blur = self._add_gauss_blur(hr_uint8)
            hr_blur = hr_blur / 127.5 - 1.
            hr_blur_sal_concat = np.concatenate((hr, hr_blur, sal), axis=2)
            hr_blur_sal_concat = hr_blur_sal_concat[0:sr_image_size, 0:sr_image_size]
            hr_img = np.expand_dims(hr_blur_sal_concat[:,:,0], axis = 2)
            hr_img_blur = np.expand_dims(hr_blur_sal_concat[:,:,1], axis = 2)
            hr_sal = np.expand_dims(hr_blur_sal_concat[:,:,2], axis = 2)
            lr_img = self._imresize(hr_img_blur, lr_image_size)
            lr_sal = self._imresize(hr_sal, lr_image_size, range=[0, 1])
            return hr_img, hr_sal, lr_img, lr_sal, hr_img_blur

        elif self.degradation_model == "BI":
            # standart degratdation model
            lr_img = self._imresize(hr_img, lr_image_size)
            return hr_img, hr_sal, lr_img, lr_sal, hr_img