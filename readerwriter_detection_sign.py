from readerwriter_detection_base import KReaderWriterBase
import csv
import os
import tensorflow as tf
import cv2
import csv
import hashlib
from data_type_tf import data_feature

from object_detection.utils import label_map_util
from common.data_extraction import DataExtraction


class KReaderWriterSign(KReaderWriterBase):
    def __init__(self):
        slim_example_decoder = tf.contrib.slim.tfexample_decoder

        self.keys_to_features = {
            'height':   tf.FixedLenFeature((), tf.int64, 1),
            'width': tf.FixedLenFeature((), tf.int64, 1),
            'depth': tf.FixedLenFeature((), tf.int64, 3),
            'label': tf.VarLenFeature(tf.int64),
            'image_raw': tf.FixedLenFeature((), tf.string, default_value=''),
            'format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'bbox/xmin': tf.VarLenFeature(tf.float32),
            'bbox/ymin': tf.VarLenFeature(tf.float32),
            'bbox/xmax': tf.VarLenFeature(tf.float32),
            'bbox/ymax': tf.VarLenFeature(tf.float32),
        }

        self.items_to_handlers = {
            'image_raw': slim_example_decoder.Image(image_key='image_raw', format_key='format', channels=3),
            'boxes': (slim_example_decoder.BoundingBox(['ymin', 'xmin', 'ymax', 'xmax'], 'bbox/')),
            'label': (slim_example_decoder.Tensor('label')),
        }

        self.acceptExtention = ['ppm']

        self.input_reader_config = DataExtraction.dict_2_obj({
            "num_epochs": None,
            "num_readers": 1,
            "shuffle": None,
            "queue_capacity": 10,
            "min_after_dequeue": 100,
        })

    def get_bounding_boxes(self, image_name):
        pass

    def _extract_features(self, image_path):
        print(image_path)
        image_array = cv2.imread(image_path)

        height = image_array.shape[0]
        width = image_array.shape[1]

        image_file_name = image_path.split('/')[-1]

        image_name = image_file_name.split('.')[0]

        image_file_jpg = os.path.join(
            "/media/kum/DATA/export_sign/raw/trans", image_name + ".jpg")

        cv2.imwrite(image_file_jpg, image_array)

        image_byte = tf.gfile.FastGFile(image_file_jpg, mode="rb").read()

        key = hashlib.sha256(image_byte).hexdigest()

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        truncated = []
        poses = []
        difficult_obj = []

        label_map_dict = {}
        for k, v in label_map_util.get_label_map_dict(self.label_map_path).iteritems():
            label_map_dict[v] = k

        with open('/media/kum/DATA/export_sign/raw/data/gt.txt') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:

                if image_file_name == row[0]:
                    difficult_obj.append(0)
                    xmin.append(float(row[1]) / width)
                    ymin.append(float(row[2]) / height)
                    xmax.append(float(row[3]) / width)
                    ymax.append(float(row[4]) / height)

                    classes_text.append(
                        label_map_dict[int(row[5])].encode('utf8'))
                    classes.append(int(row[5]))

                    truncated.append(0)
                    poses.append(''.encode('utf8'))

        features = {
            'height': data_feature.int64_feature(height),
            'width': data_feature.int64_feature(width),
            'depth': data_feature.byte_feature(image_file_name.encode('utf8')),
            'image_raw': data_feature.byte_feature(image_byte),
            'bbox/xmin': data_feature.float_list_feature(xmin),
            'bbox/xmax': data_feature.float_list_feature(xmax),
            'bbox/ymin': data_feature.float_list_feature(ymin),
            'bbox/ymax': data_feature.float_list_feature(ymax),
            'label': data_feature.int64_list_feature(classes),
        }

        return features

    def translate_label_map(self, file):
        with open('/media/kum/9F3C-DEF0/data/id.txt') as csvfile:
            reader = csv.reader(csvfile, delimiter='=')
            with open('/media/kum/9F3C-DEF0/data/sign_label_map.pbtxt', "w") as newcsvfile:
                for row in reader:
                    newclass = "item {id: %s, name: '%s'}  \n" % (
                        row[0], row[1])
                    newcsvfile.writelines(newclass)
