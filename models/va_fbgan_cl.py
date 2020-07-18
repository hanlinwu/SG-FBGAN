from models.va_fbgan import VA_FBGAN
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, ElementwiseLayer, ConcatLayer, BatchNormLayer, DeConv2d
from utils.utils import p_snr, compute_charbonnier_loss, psnr
from tqdm import tqdm
import numpy as np
from cv2 import imwrite

class VA_FBGAN_CL(VA_FBGAN):
    def __init__(self, config):
        super(VA_FBGAN_CL, self).__init__(config)

    def build_model(self, config):
        self.is_train = True #self.config.is_train

        ###========================== INPUT OPTS ==========================###
        self.lr_img = tf.placeholder('float32', [None, None, None, self.in_channel], name='lr_img')
        self.hr_img = tf.placeholder('float32', [None, None, None, self.in_channel], name='hr_img')

        self.lr_sal = tf.placeholder('float32', [None, None, None, 1], name = 'lr_sal')
        self.hr_sal = tf.placeholder('float32', [None, None, None, 1], name = 'hr_sal')
        self.hr_img_intermediate = tf.placeholder('float32', [None, None, None, 1], name='hr_img_intermediate')

        ###========================== GENERATOR ==========================###
        net_g = self._generator(self.lr_img, self.lr_sal, self.upscale, D = config.MODEL.D, C = config.MODEL.C, m = config.MODEL.channel_num, reuse = False, T_int = config.MODEL.itr, is_train = self.is_train)

        ###========================== DISCRIMINATOR ==========================###
        logits_real_sa = self._discriminator(self.hr_img, self.hr_sal, 'sa', is_train=self.is_train, reuse=False)
        logits_fake_sa_list = [self._discriminator(_.outputs, self.hr_sal, 'sa', is_train=self.is_train, reuse=True) for _ in net_g[1:]]

        logits_real_no = self._discriminator(self.hr_img, 1 - self.hr_sal, 'no', is_train=self.is_train, reuse=False)
        logits_fake_no_list = [self._discriminator(_.outputs,  1 - self.hr_sal, 'no', is_train=self.is_train, reuse=True) for _ in net_g[1:]]

        logits_real = self._discriminator(self.hr_img, name = 'global', is_train=self.is_train, reuse=False)
        logits_fake_list = [self._discriminator(_.outputs, name = 'global', is_train=self.is_train, reuse=True) for _ in net_g[1:]]

        ###========================== COMPUTE G LOSS ==========================###
        # g_loss / L_1 loss
        L1_loss_list_1 = [compute_charbonnier_loss(self.hr_img_intermediate, _.outputs, is_mean=True) for _ in net_g[0:1]]
        L1_loss_list_2 = [compute_charbonnier_loss(self.hr_img, _.outputs, is_mean=True) for _ in net_g[1:]]
        L1_loss_list = L1_loss_list_1 + L1_loss_list_2
        self.L1_loss = tf.reduce_mean(L1_loss_list)

        # g_loss / g_gan_loss
        if 'sa' in config.MODEL.g_gan_loss_whattouse:
            g_gan_loss_sa = tf.reduce_mean([tl.cost.sigmoid_cross_entropy(l, tf.ones_like(l)) for l in logits_fake_sa_list])
        else :
            g_gan_loss_sa = tf.constant(0, tf.float32)
        if 'no' in config.MODEL.g_gan_loss_whattouse:
            g_gan_loss_no = tf.reduce_mean([tl.cost.sigmoid_cross_entropy(l, tf.ones_like(l)) for l in logits_fake_no_list])
        else :
            g_gan_loss_no = tf.constant(0, tf.float32)
        if 'global' in config.MODEL.g_gan_loss_whattouse:
            g_gan_loss_global = tf.reduce_mean([tl.cost.sigmoid_cross_entropy(l, tf.ones_like(l)) for l in logits_fake_list])
        else :
            g_gan_loss_global = tf.constant(0, tf.float32)

        g_gan_loss = config.MODEL.g_gan_loss_alpha * (g_gan_loss_sa + g_gan_loss_no + g_gan_loss_global) * 1./len(config.MODEL.g_gan_loss_whattouse)

        self.g_loss = self.L1_loss + g_gan_loss

        ###========================== COMPUTE D LOSS ==========================###
        d_loss1_sa = tl.cost.sigmoid_cross_entropy(logits_real_sa, tf.ones_like(logits_real_sa), name='d1_sa')
        d_loss2_sa = tf.reduce_mean([tl.cost.sigmoid_cross_entropy(l, tf.zeros_like(l)) for l in logits_fake_sa_list], name='d2_sa')

        d_loss1_no = tl.cost.sigmoid_cross_entropy(logits_real_no, tf.ones_like(logits_real_no), name='d1_no')
        d_loss2_no = tf.reduce_mean([tl.cost.sigmoid_cross_entropy(l, tf.zeros_like(l)) for l in logits_fake_no_list], name='d2_no')

        d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
        d_loss2 = tf.reduce_mean([tl.cost.sigmoid_cross_entropy(l, tf.zeros_like(l)) for l in logits_fake_list], name='d2')

        d_loss_sa = d_loss1_sa + d_loss2_sa
        d_loss_no = d_loss1_no + d_loss2_no
        d_loss_global = d_loss1 + d_loss2

        self.d_loss = d_loss_sa + d_loss_no + d_loss_global
        self.output = net_g[-1].outputs