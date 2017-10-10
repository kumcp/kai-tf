import tensorflow as tf



class KDetection(object):
    NUM_CLASSES = 1

    def __init__(self):
        pass

    def train(self,train_dir=None):
        if train_dir==None:
            train_dir=self.data_dir

    def eval(self):
        pass

    def detect(self):
        pass