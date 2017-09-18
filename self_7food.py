
import os.path
import tensorflow as tf
from classification import KClassification

class K7FoodClassification(KClassification):

    def __init__(self):

        self.set_batch_size(32)
        self.log_dir = "./log"
        self.data_dir = os.path.join(os.path.realpath(__file__),'../cifar10')
        self.eval_log_dir = "./log/eval"
        self.num_test = 30000
        self.force_start = True
        self.sess = tf.InteractiveSession()
        self.num_run = 200
        self.ini_lr = 0.02
        self.decay_rate = 0.99
        self.decay_step = 100000
        self.crop_size=[80,80]

    def set_batch_size(self,batch_size):
        self.BATCH_SIZE = batch_size

