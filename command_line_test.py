import sys

sys.dont_write_bytecode = True

from readerwriter_detection_sign import KReaderWriterSign
import os.path
from self_detect import SignDetection
import tensorflow as tf


path = '/media/kum/DATA/export_sign/raw/data'
export_dir = '/media/kum/DATA/export_sign/'
label_map_path = '/media/kum/DATA/export_sign/raw/data/sign_label_map.pbtxt'

# krw = KReaderWriterSign()

# krw.set_label_map_path(label_map_path)
# krw.convert_real_image_to_tfrecord(path, export_dir, "")


mydtc = SignDetection()
train_dir = export_dir + "train/1.tfrecord"
eval_dir = export_dir + "test/"
mydtc.train(train_dir)
# mydtc.eval(train_dir)
