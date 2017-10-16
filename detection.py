import tensorflow as tf

from architectures.ssd.model import ssd_model
from readerwriter_detection_sign import KReaderWriterSign


class KDetection(object):
    NUM_CLASSES = 1

    def __init__(self, extend_file_name, extend_class):
        pass

    def train(self, train_dir=None):
        if train_dir == None:
            train_dir = self.data_dir

        input_queue = self.read_data(train_dir)

        model_fn = functools.partial(
            model_builder.build,
            model_config=model_config,
            is_training=True)

        model_fn = functools.partial(ssd_model, is_training=True)

        create_losses([input_queue], create_model_fn=model_fn)
        # clones = model_deploy.create_clones(deploy_config, model_fn, [input_queue])
        ################################

        images, labels = self.read_data(train_dir)

    def read_data(self, data_dir):

        krw = KReaderWriterSign()

        return krw.train_data_tfrecord(data_dir, self.train_config)

    def eval(self):
        pass

    def detect(self):
        pass
