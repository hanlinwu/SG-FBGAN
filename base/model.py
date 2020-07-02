import tensorflow as tf
from utils.logger import Logger

class BaseModel:
    def __init__(self, config):
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, filename="model"):
        print("Saving model...")
        self.saver.save(self.sess, self.config.checkpoint_dir + '/{0}.ckpt'.format(filename), self.cur_epoch_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        # just copy the following line in your child class
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def build_model(self):
        raise NotImplementedError

    def compile(self, data):
        # create tensorflow session
        self.sess = tf.Session()
        # creat data generator
        self.data = data
        # create tensorboard logger
        self.logger = Logger(self.sess, self.config)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init)
        self.fit = lambda : self.__fit()
        # init the saver 
        self.init_saver()

    def __fit(self):
        for _ in range(self.cur_epoch_tensor.eval(self.sess), self.config.TRAIN.n_epoch + 1, 1):
            self.train_epoch()
            self.sess.run(self.increment_cur_epoch_tensor)

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError