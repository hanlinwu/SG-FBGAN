from base.model import BaseModel
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, ElementwiseLayer, ConcatLayer, BatchNormLayer, DeConv2d
from utils.utils import p_snr, compute_charbonnier_loss, psnr
from tqdm import tqdm
import numpy as np
import time
from cv2 import imwrite

class VA_FBGAN(BaseModel):
    def __init__(self, config):
        super(VA_FBGAN, self).__init__(config)
        self.upscale = config.MODEL.upscale
        self.in_channel = config.MODEL.in_channel
        #Adam Parameters Setting
        self.lr_init = tf.Variable(self.config.TRAIN.lr_init, trainable= False, name = "lr_init")
        self.beta1 = config.TRAIN.beta1
        self.build_model(config)
        self._init_train_options()

    def _RDN_block(self, n, m_filter, D = 2, C = 3, reuse = False, is_train = False, scope_name = ''):
        w_init = tf.random_normal_initializer(stddev=0.005)
        g_init = tf.random_normal_initializer(1., 0.02)
        lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

        with tf.variable_scope("RDN_block_{0}".format(scope_name), reuse=reuse):
            for i in range(D):
                xx = n
                rdb_concat = []
                for j in range(C):
                    n = Conv2d(n, 2 * m_filter, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='sa_Conv3_3_%s_%s'%(i,j))
                    n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='sa_n64s1/b1/%s_%s'%(i,j))
                    aa = n
                    for s in range(len(rdb_concat)):
                        n = ConcatLayer([n, rdb_concat[s]], 3, name='sa_concat%s_%s_%s' % (i,j,s))
                    rdb_concat.append(aa)
                # local feature fusion
                n = Conv2d(n, 2 * m_filter, (1, 1), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='sa_Conv1_1_%s_%s' % (i, j))
                n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='sa_n64s1/b1/%s' % i)
                # n = n + xx
                n = ElementwiseLayer([n, xx], combine_fn=tf.add, name='sa_add_layers_%s'%i)
                # local residual learning
                if i != 0:
                    rdb_output = ConcatLayer([rdb_output,n], 3, name='sa_concat%s' % (i))
                else:
                    rdb_output = n

            rdb_output = Conv2d(rdb_output, m_filter, (1,1), (1,1), act = lrelu, padding = 'SAME', W_init=w_init)
        return rdb_output

    def _Feedback_block(self, F_in, F_out, smap, nmap, m_filter, D = 2, C = 3, reuse = False, is_train = False):
        # calculate saliency map and no-saliency map
        with tf.variable_scope("Feedback_block", reuse = reuse):
            # concat input and output of last step
            input = ConcatLayer([F_in, F_out], 3, name='feed_block_concat')

            # generate inputs for RDB-blocks
            input_sa = ElementwiseLayer([input, smap], tf.multiply, name='gate_con_sa')
            input_no = ElementwiseLayer([input, nmap], tf.multiply, name='gate_con_no')

            out_sa = self._RDN_block(input_sa, m_filter, D = D, C = C, reuse = False, is_train = is_train, scope_name='sa')
            out_no = self._RDN_block(input_no, m_filter, D = D, C = C, reuse = False, is_train = is_train, scope_name='no')

            out = ElementwiseLayer([out_sa, out_no], combine_fn=tf.add, name='add')

        return out

    def _generator(self, input_image, s_map, upscale,  m = 64, D = 2, C = 3, reuse = False, T_int = 4, is_train = False):
        def Recons_block(F_in, reuse = False, scope_name = ''):
            lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
            with tf.variable_scope("Recons_block_{0}".format(scope_name), reuse = reuse):
                # stride control the magnification times
                HR_res = DeConv2d(F_in, n_filter= 32, filter_size= (upscale, upscale), strides= (upscale, upscale), padding='SAME', act= lrelu, name = 'de_convolution')
                HR = Conv2d(HR_res, self.in_channel, (3, 3), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='conv_3x3')
            return HR

        s_map = tf.tile(s_map, [1, 1, 1, 2*m])
        n_map = 1-s_map
        smap = InputLayer(s_map, name='smap')
        nmap = InputLayer(n_map, name='nmap')
        w_init = tf.random_normal_initializer(stddev=0.005)
        n = InputLayer(input_image, name='in')
        lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
        with tf.variable_scope("Feedback_saliency_generator", reuse=reuse):
            F_in = Conv2d(n, 4*m, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='F_in_3x3')
            F_in = Conv2d(F_in, m, (1, 1), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='F_in_1x1')
            HR_outs = []
            # recurrent blocks
            for i in range(T_int):
                F_out = FB_out if i > 0 else F_in
                FB_out = self._Feedback_block(F_in, F_out, smap, nmap, m, D = D, C = C, reuse = tf.AUTO_REUSE, is_train = is_train)
                HR_outs.append(Recons_block(FB_out, scope_name = 'all', reuse = tf.AUTO_REUSE))
        return HR_outs

    def _discriminator(self, input_images, bin_map = None, name = '', is_train=True, reuse=False):
        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = None  # tf.constant_initializer(value=0.0)
        df_dim = 32
        lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
        with tf.variable_scope("SRGAN_d_pathch_new_{0}".format(name), reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            net_in_bin = InputLayer(input_images, name='sa_input/images')
            if bin_map is not None and name not in ('sa','no'):
                raise RuntimeError('Error : unrecognized name')
            if bin_map is not None:
                binmap = InputLayer(bin_map, name='sa_input/binmap')
                net_in_bin = ElementwiseLayer([net_in_bin, binmap],tf.multiply, name='gate_con_sa')
            net_h0 = Conv2d(net_in_bin, df_dim, (4, 4), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='sa_h0/c')
            net_h1 = Conv2d(net_h0, df_dim , (4, 4), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='sa_h1/c')
            net_h1 = MaxPool2d(net_h1, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='sa_h1/mp')
            net_h2 = Conv2d(net_h1, df_dim , (4, 4), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='sa_h2/c')
            net_h2 = MaxPool2d(net_h2, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='sa_h2/mp')
            net_h3 = Conv2d(net_h2, df_dim , (4, 4), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='sa_h3/c')
            net_h3 = MaxPool2d(net_h3, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='sa_h2/mp')
            net_h5 = Conv2d(net_h3, 1, (4, 4), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='sa_hout/c')
            logits = net_h5.outputs
            net_h5.sig = tf.nn.sigmoid(logits)
            net_h5.outputs = logits
        return logits

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
        logits_fake_sa_list = [self._discriminator(_.outputs, self.hr_sal, 'sa', is_train=self.is_train, reuse=True) for _ in net_g]

        logits_real_no = self._discriminator(self.hr_img, 1 - self.hr_sal, 'no', is_train=self.is_train, reuse=False)
        logits_fake_no_list = [self._discriminator(_.outputs,  1 - self.hr_sal, 'no', is_train=self.is_train, reuse=True) for _ in net_g]

        logits_real = self._discriminator(self.hr_img, name = 'global', is_train=self.is_train, reuse=False)
        logits_fake_list = [self._discriminator(_.outputs, name = 'global', is_train=self.is_train, reuse=True) for _ in net_g]

        ###========================== COMPUTE G LOSS ==========================###
        # g_loss / L_1 loss
        L1_loss_list = [compute_charbonnier_loss(self.hr_img, _.outputs, is_mean=True) for _ in net_g]
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

    def _init_train_options(self):
        g_vars = tl.layers.get_variables_with_name('Feedback_saliency_generator', True, True)
        d_vars_sa = tl.layers.get_variables_with_name('SRGAN_d_pathch_new_sa', True, True)
        d_vars_no = tl.layers.get_variables_with_name('SRGAN_d_pathch_new_no', True, True)
        d_vars_global = tl.layers.get_variables_with_name('SRGAN_d_pathch_new_global', True, True)

        d_vars = []
        if 'sa' in self.config.MODEL.g_gan_loss_whattouse:
            d_vars += d_vars_sa
        if 'no' in self.config.MODEL.g_gan_loss_whattouse:
            d_vars += d_vars_no
        if 'global' in self.config.MODEL.g_gan_loss_whattouse:
            d_vars += d_vars_global
            
        # Define train opts
        
        self.train_opt_init = tf.train.AdamOptimizer(self.lr_init, beta1=self.beta1).minimize(self.L1_loss, var_list=g_vars)
        self.train_opt_g = tf.train.AdamOptimizer(self.lr_init,beta1 =self.beta1).minimize(self.g_loss, var_list=g_vars)
        self.train_opt_d = tf.train.AdamOptimizer(self.lr_init, beta1=self.beta1).minimize(self.d_loss, var_list=d_vars)

        # Define update lr opt
        cur_epoch = self.cur_epoch_tensor
        lr_init = self.config.TRAIN.lr_init
        lr_decay = self.config.TRAIN.lr_decay
        lr_decay_every = self.config.TRAIN.lr_decay_every
        new_lr = lr_init * (tf.pow(lr_decay, tf.cast(cur_epoch // lr_decay_every, tf.float32)))
        self.update_lr_tensor = tf.assign(self.lr_init, new_lr)

    def train_epoch(self):
        # the current opoch
        cur_epoch = self.cur_epoch_tensor.eval(self.sess)

        # initialize iterator
        self.sess.run(self.data.train.initializer)
        self.get_one_batch = self.data.train.get_next()

        # update lr
        if cur_epoch % self.config.TRAIN.lr_decay_every == 0:
            print("learning rate updated to [%e]." % self.lr_init.eval(self.sess))
            self.sess.run(self.update_lr_tensor)

        # train_init
        if cur_epoch < self.config.TRAIN.n_epoch_init :
            loop = tqdm(range(self.data.batch_num))
            loop.set_description("Init Epoch [%2d/%2d]" % (cur_epoch, self.config.TRAIN.n_epoch_init))
            L1_losses = []
            for _ in loop:
                L1_loss = self._train_step_init()
                L1_losses.append(L1_loss)
            L1_loss = np.mean(L1_losses)
            g_loss = np.array(0)
            d_loss = np.array(0)

        # train_gan
        else:
            loop = tqdm(range(self.data.batch_num))
            loop.set_description("GAN Epoch [%2d/%2d]" % (cur_epoch, self.config.TRAIN.n_epoch))
            L1_losses = []
            g_losses = []
            d_losses = []
            for _ in loop:
                d_loss, g_loss, L1_loss = self._train_step_gan()
                g_losses.append(g_loss)
                d_losses.append(d_loss)
                L1_losses.append(L1_loss)
            g_loss = np.mean(g_losses)
            d_loss = np.mean(d_losses)
            L1_loss = np.mean(L1_losses)

        psnr_on_valid, samples = self._quick_evaluate()
        # youhua 
        summaries_dict = {
            'L1_loss': L1_loss,
            'g_loss': g_loss,
            'd_loss': d_loss,
            "psnr" : psnr_on_valid,
            "sample" : samples
        }

        self.logger.summarize(cur_epoch, summaries_dict=summaries_dict)
        if cur_epoch % self.config.TRAIN.save_model_each == 0:
            self.save()
        
    def _train_step_init(self):
        hr_img, hr_sal, lr_img, lr_sal, hr_img_intermediate = self.sess.run(self.get_one_batch)
        feed_dict = {
            self.lr_img: lr_img,
            self.hr_img: hr_img, 
            self.lr_sal: lr_sal,
            self.hr_sal: hr_sal,
            self.hr_img_intermediate : hr_img_intermediate
            }
        _, L1_loss = self.sess.run([self.train_opt_init, self.L1_loss], feed_dict=feed_dict)
        return L1_loss

    def _train_step_gan(self):
        hr_img, hr_sal, lr_img, lr_sal, hr_img_intermediate = self.sess.run(self.get_one_batch)
        feed_dict = {
            self.lr_img: lr_img,
            self.hr_img: hr_img, 
            self.lr_sal: lr_sal,
            self.hr_sal: hr_sal,
            self.hr_img_intermediate : hr_img_intermediate
            }

        # update D
        _, d_loss, L1_loss = self.sess.run([self.train_opt_d, self.d_loss, self.L1_loss], feed_dict)

        # update G
        _, g_loss = self.sess.run([self.train_opt_g, self.g_loss], feed_dict)
        return d_loss, g_loss, L1_loss

    def get_quick_eval_data(self):
        self.sess.run(self.data.valid.initializer)
        self.get_one_batch = self.data.valid.get_next()

    def _quick_evaluate(self):
        self.sess.run(self.data.valid.initializer)
        self.get_one_batch = self.data.valid.get_next()
        loop = range(min(self.data.valid_length, 5))
        psnrs = []
        outs = []
        for _ in loop:
            hr_img, _, lr_img, lr_sal, _ = self.sess.run(self.get_one_batch)
            out, _ = self.predict(lr_img, lr_sal)
            hr_img = ((hr_img + 1)*127.5).astype(np.uint8)
            outs.append(out)
            psnrs.append(psnr(out, hr_img))

        ps_mean_value = np.mean(np.array(psnrs))
        return ps_mean_value, np.concatenate(outs, axis = 0)

    def evaluate(self):
        self.sess.run(self.data.valid.initializer)
        self.get_one_batch = self.data.valid.get_next()
        loop = range(self.data.valid_length)
        psnrs = []
        cal_times = []
        save_dir = self.config.evaluate_dir
        for idx in loop:
            hr_img, _, lr_img, lr_sal, _ = self.sess.run(self.get_one_batch)
            out, cal_time = self.predict(lr_img, lr_sal)
            hr_img = ((hr_img + 1)*127.5).astype(np.uint8)
            psnrs.append(psnr(out, hr_img))
            cal_times.append(cal_time)
            print(cal_time)
            # save images
            imwrite(save_dir + '/' + self.data.valid_names[idx] + '.tif', np.squeeze(out))
        ps_mean_value = np.mean(np.array(psnrs))
        cal_time_mean = np.mean(np.array(cal_times))
        print("mean psnr" , ps_mean_value)
        print("mean time" , cal_time_mean)
        return ps_mean_value

    def predict(self, lr_img, lr_sal):
        feed_dict = {self.lr_img : lr_img, self.lr_sal : lr_sal}
        start = time.time()
        out = self.sess.run(self.output, feed_dict = feed_dict)
        end = time.time()
        out = ((out + 1) * 127.5).astype(np.uint8)
        return out, end-start