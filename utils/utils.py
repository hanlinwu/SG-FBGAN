import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *

import imageio
import cv2
import numpy as np
from .pca_conv import PCA_CONV, Info
from tensorflow.python.client import device_lib

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='', help='optionp file location')
    parser.add_argument('--gpu', type=str, default='', help='gpu to use')
    args = parser.parse_args()
    return args

def check_available_gpus():
    local_devices = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
    gpu_num = len(gpu_names)
    print('{0} GPUs are detected : {1}'.format(gpu_num, gpu_names))
    return gpu_num

def get_imgs_fn(file_name, path, flag):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    if flag == 'Image':
        im = cv2.imread(path + file_name)
        im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
        return np.expand_dims(((im_lab[:, :, 0] / 127.5) - 1.), axis=2)
    elif flag == 'Saliency':
        im = imageio.imread(path + file_name) / 255.0
        return im / im.max()
    elif flag == 'DeepSaliency':
        im = cv2.imread(path + file_name)
        im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)/255.0
        return im_lab/im_lab.max()
    else:
        raise AssertionError("Reading flag is not given ('Image' or 'Saliency').")

def crop_sub_imgs_fn(x, w, h, is_random=True):
    temp = crop(x, wrg=w, hrg=h, is_random=is_random)
    return temp

def downsample_fn(x, w, h):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = (x + 1.) * 127.5
    if len(x.shape) == 2:
        x = np.expand_dims(x, axis=2)
    x = imresize(x, size=[w, h], interp='bicubic', mode=None)
    x = x / 127.5 - 1
    return x

def upsample_fn(x, w, h):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = (x + 1.) * 127.5
    if len(x.shape) == 2:
        x = np.expand_dims(x, axis=2)
    x = imresize(x, size=[w, h], interp='bicubic', mode=None)
    x = x / 127.5 - 1
    return x

def downsample_scale(x, w, h, scale):
    new_w = w // scale
    new_h = h // scale
    x = (x + 1.) * 127.5
    x = np.squeeze(x)
    x = x[0: new_w * scale, 0 : new_h *scale]
    if len(x.shape) == 2:
        x = np.expand_dims(x, axis=2)
    x = imresize(x, size=[new_w, new_h], interp='bicubic', mode=None)
    x = x / 127.5 - 1.
    return x

def downsample_scale_saliency(x, w, h, scale):
    new_w = w // scale
    new_h = h // scale
    x = x * 255.0
    x = np.squeeze(x)
    x = x[0: new_w * scale, 0 : new_h *scale]
    if len(x.shape) == 2:
        x = np.expand_dims(x, axis=2)
    x = imresize(x, size=[new_w, new_h], interp='bicubic', mode=None)
    x = x / 255.0
    return x

def psnr(x, y, maximum=255.0):
    return 20 * np.log10(maximum / np.sqrt(np.mean((1. *  x - 1. * y) ** 2)))

def n_mse(t_target_image, generated_image, batch_size):
    t_residual_image = tf.subtract(t_target_image, generated_image)
    hyperparams = {
        'k1': 4,
        'k2': 4,
        'l1': 16,
    }
    info = Info(n_channels=1, image_h=384, image_w=384, batch_size=batch_size)
    pca_conv = PCA_CONV(t_residual_image, hyperparams, info)
    nmse = tf.reduce_sum(pca_conv.outputs)
    for ind in range(hyperparams['l1']):
        tf.summary.scalar('z_eigenvalue_%d' % ind, tf.squeeze(pca_conv.outputs[ind]))
    return nmse

def m_se(t_target_image, generated_image):
    return tl.cost.mean_squared_error(generated_image, t_target_image, is_mean=True)

def p_snr(t_target_image, generated_image):
    op1 = tf.subtract((generated_image + 1.) * 127.5, (t_target_image + 1.) * 127.5)
    op2 = tf.pow(op1, 2)
    op3 = tf.squeeze(tf.reduce_mean(op2, reduction_indices=(1, 2, 3), keep_dims=True))
    op4 = tf.sqrt(op3)
    op4_5 = np.log10(255.) - tf.divide(tf.log(op4), np.log(10.))
    psnr_mean = tf.reduce_mean(20. * op4_5)
    return psnr_mean

def gram_matrix(net):
    a = net.outputs
    b,h,w,c = a.get_shape().as_list()
    feats = tf.reshape(net.outputs, (b, h*w, c))
    feats_T = tf.transpose(feats, perm=[0,2,1])
    grams = tf.matmul(feats_T, feats) / h/w/c
    return grams

def content_loss(vgg_target_emb_54, vgg_predict_emb_54):
    a = vgg_target_emb_54.outputs
    b, h, w, c = a.get_shape().as_list()
    gram_target = gram_matrix(vgg_target_emb_54)
    gram_generated = gram_matrix(vgg_predict_emb_54)
    return 2 * tf.nn.l2_loss(gram_target - gram_generated) / 512 / 24 / 24

def compute_charbonnier_loss(tensor1, tensor2, is_mean=True):
    epsilon = 1e-6
    if is_mean:
        loss = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(tensor1,tensor2))+epsilon), [1, 2, 3]))
    else:
        loss = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(tensor1,tensor2))+epsilon), [1, 2, 3]))

    return loss
