from readerwriter_detection_base import KReaderWriterBase
import csv
import os
import tensorflow as tf


class KReaderWriter(KReaderWriterBase):
    def __init__(self):
        slim_example_decoder = tf.contrib.slim.tfexample_decoder

        self.keys_to_features = {
            'height':   tf.FixedLenFeature((), tf.int64, 1),
            'width': tf.FixedLenFeature((), tf.int64, 1),
            'depth': tf.FixedLenFeature((), tf.int64, 3),
            'label': tf.VarLenFeature(tf.int64),
            'image_raw': tf.FixedLenFeature((), tf.string, default_value=''),
            'bbox/xmin': tf.VarLenFeature(tf.float32),
            'bbox/ymin': tf.VarLenFeature(tf.float32),
            'bbox/xmax': tf.VarLenFeature(tf.float32),
            'bbox/ymax': tf.VarLenFeature(tf.float32),
        }

        self.items_to_handlers = {
            'image_raw': slim_example_decoder.Image(image_key='image_raw', shape=[slim_example_decoder.Tensor('height'),
                                                                                  slim_example_decoder.Tensor(
                                                                                      'width'),
                                                                                  slim_example_decoder.Tensor('depth')]),
            'boxes': (slim_example_decoder.BoundingBox(['ymin', 'xmin', 'ymax', 'xmax'], 'bbox/')),
            'label': (slim_example_decoder.Tensor('label')),
        }

    def get_bounding_boxes(self, image_name):
        bb_path = os.path.join(self.path, "boxes", image_name + ".csv")

        with open(bb_path, 'rb') as file:
            reader = csv.reader(file, delimiter=",")
            bbxmin = []
            bbymin = []
            bbxmax = []
            bbymax = []
            label = []
            hidden = []
            for line in reader:
                bbxmin.append(int(line[0]))
                bbymin.append(int(line[1]))
                bbxmax.append(int(line[2]))
                bbymax.append(int(line[3]))
                label.append(int(line[4]))
                item_in_row = []
                for hidden_item in range(5, len(line)):
                    item_in_row.append(line[hidden_item])
                if len(item_in_row) > 0:
                    hidden.append(item_in_row)

        result = {
            "bounding_box_xmin": bbxmin,
            "bounding_box_ymin": bbymin,
            "bounding_box_xmax": bbxmax,
            "bounding_box_ymax": bbymax,
            "label_int": label,
        }

        if len(hidden) > 0:
            result["hidden"] = hidden

        return result
