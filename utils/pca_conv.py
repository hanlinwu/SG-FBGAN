import tensorflow as tf
from tensorflow.contrib import image
import numpy as np

class PCA_CONV:
    def __init__(self, init_image_batch, hyperparams, info):
        self._image_batch = init_image_batch
        # tf.summary.image('input', self._image_batch, max_outputs=5)

        k1 = hyperparams['k1']
        k2 = hyperparams['k2']
        l1 = hyperparams['l1']

        with tf.name_scope("extract_patches1"):
            self.patches1 = tf.extract_image_patches(self._image_batch, [1, k1, k2, 1], [1, k1, k2, 1], [1, 1, 1, 1], padding='SAME', name='patches')
            self.patches1 = tf.reshape(self.patches1, [-1, k1 * k2 * info.N_CHANNELS], name='patches_shaped')
            self.numofpatches = self.patches1.get_shape()[0].value
            # TODO: figure out how to unvectorize for multi-channel images
            # self.patches1 = tf.reshape(self.patches1, [-1, info.N_CHANNELS,  k1 * k2], name='patches_shaped')
            # self.patches1 = tf.transpose(self.patches1, [0, 2, 1])
            # self.zero_mean_patches1 = self.patches1 - tf.reduce_mean(self.patches1, axis=1, keep_dims=True, name='patch_means')
            self.zero_mean_patches1 = self.patches1
            x1 = tf.transpose(self.zero_mean_patches1, [1, 0])
            x1_trans = self.zero_mean_patches1
            self.patches_covariance1 = tf.matmul(x1, x1_trans, name='patch_covariance')

        with tf.name_scope("eignvalue_decomposition1"):
            self.x_eig_vals1, self.x_eig1 = tf.self_adjoint_eig(self.patches_covariance1, name='x_eig')

            self.top_x_eig1_ori = tf.reverse(self.x_eig1, axis=[-1])[:, 0:l1]
        #     self.top_x_eig1 = tf.transpose(tf.reshape(self.top_x_eig1_ori, [info.N_CHANNELS, k1, k2, l1]), [2, 1, 0, 3])
        #     self.top_x_eig2 = tf.transpose(image.rotate(tf.reshape(self.top_x_eig1_ori, [info.N_CHANNELS, k1, k2, l1]), np.pi), [2, 1, 3, 0])
        #
            self.top_x_eig_vals1 = tf.expand_dims(tf.reverse(self.x_eig_vals1, axis=[-1])[0:l1], axis=1)
        #     # self.top_x_eig_vals1 = self.x_eig_vals1[0:l1]
        #     self.filt1_viz = tf.transpose(self.top_x_eig1, [3, 0, 1, 2])
        #     tf.summary.image('filt1', self.filt1_viz, max_outputs=l1)
        #
        # with tf.name_scope("convolution1"):
        #     self.conv1 = tf.nn.conv2d(self._image_batch, self.top_x_eig1, [1, 1, 1, 1], padding='SAME')
        #
        #     self.conv1 = tf.transpose(self.conv1, [3, 0, 1, 2])
        #     # conv1 is now (l1, batch_size, img_w, img_h)
        #     self.conv1_batch = tf.expand_dims(tf.reshape(self.conv1, [-1, info.IMAGE_W, info.IMAGE_H]), axis=3)
        #     # conv1 batch is (l1 * batch_size, img_w, img_h)
        #
        #     tf.summary.image('conv1', self.conv1_batch, max_outputs=l1)
        #
        # with tf.name_scope("normalization_of_convolution"):
        #     self.conv1_flatten = tf.reshape(self.conv1, [l1, info.batch_size * info.IMAGE_W * info.IMAGE_H])
        #     self.eigen_vals = tf.tile(self.top_x_eig_vals1, [1, info.batch_size * info.IMAGE_W * info.IMAGE_H])
        #     self.conv1_div_vals = tf.divide(self.conv1_flatten, tf.sqrt(tf.sqrt(self.eigen_vals)))
        #     self.conv1_output = tf.transpose(tf.reshape(self.conv1_div_vals, [l1, info.batch_size, info.IMAGE_W, info.IMAGE_H]), [1, 2, 3, 0])
        #     self.outputs = tf.nn.conv2d(self.conv1_output, self.top_x_eig2, [1, 1, 1, 1], padding='SAME')
        #     # self.outputs = self.conv1_flatten

        # We proved that mse_loss = Sum(eigen_vals), thus we do not need any convolutions ops. Modified at 5.24 14:38
        with tf.name_scope('MSE_Scaling_Op'):
            self.eigen_val_sum = tf.reduce_sum(self.top_x_eig_vals1)
            self.outputs = tf.sqrt(1.0 / self.numofpatches * self.top_x_eig_vals1)

        with tf.name_scope('Decomposition_maps'):
            self.decompatches = []
            for i in range(k1 * k2):
                self.decompatches.append(tf.matmul(tf.matmul(self.zero_mean_patches1, tf.expand_dims(self.top_x_eig1_ori[:, i], axis=1))
                                                   , tf.expand_dims(self.top_x_eig1_ori[:, i], axis=0)))

    def set_input_tensor(self, image_batch):
        self._image_batch = image_batch


class Info:
    def __init__(self, n_channels, image_h, image_w, batch_size):
        self.N_CHANNELS = n_channels
        self.IMAGE_H = image_h
        self.IMAGE_W = image_w
        self.batch_size = batch_size