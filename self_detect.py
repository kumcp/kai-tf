
import os.path
import tensorflow as tf
from detectuib import KDetection
from layers import KLayer

class KObjectDetection(KDetection):

    def __init__(self):

        self.set_batch_size(32)
        self.log_dir = "./log"
               
        self.num_test = 30000
        self.force_start = False
        self.sess = tf.InteractiveSession()
        self.num_run = 10000
        self.ini_lr = 0.01
        self.decay_rate = 0.9
        self.decay_step = 100000
        self.crop_size=[300,300]
        self.model_name = "detection_test_061017"
        self.eval_log_dir = "./log/"+self.model_name
        self.data_dir = "train_dir"

    def set_batch_size(self,batch_size):
        self.BATCH_SIZE = batch_size
        KLayer.set_batchsize(batch_size)

    def model(self,images,reuse=None,is_training=True):
        pass