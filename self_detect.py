
import os.path
import tensorflow as tf
from detection import KDetection
from layers import KLayer
from common.data_extraction import DataExtraction


class SignDetection(KDetection):

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
        self.crop_size = [300, 300]
        self.model_name = "detection_test_061017"
        self.eval_log_dir = "./log/" + self.model_name
        self.data_dir = "train_dir"

        self.extend_file_name = "readerwriter_detection_sign"
        self.extend_class = "KReaderWriterSign"

        self.train_config = DataExtraction.dict_2_obj({
            "batch_queue_capacity": 1,
            "num_batch_queue_threads": 1,
            "prefetch_queue_capacity": 1,
        })

    def set_batch_size(self, batch_size):
        self.BATCH_SIZE = batch_size
        KLayer.set_batchsize(batch_size)

    def model(self, images, reuse=None, is_training=True):
        pass
