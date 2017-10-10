import tensorflow as tf
import cv2
import os
import random

import six
import abc

from object_detection.utils import label_map_util
from datamanipulating import KDataManipulating
from data_type_tf import data_feature


@six.add_metaclass(abc.ABCMeta)
class KReaderWriterBase(object):
    
    def __init__(self):
        pass
        
    def set_size(self,size=[],label_bytes=1,isImage=True):

        self.size = size
        self.label_bytes = label_bytes

        if isImage:
            self.height, self.width, self.depth = self.size

            image_bytes = 1
            for i in size:
                if i<=0:
                    return False
                image_bytes *= i
            self.image_bytes = image_bytes

            # Number of bytes that needed for a record (a image + its label)
            self.record_bytes = self.image_bytes + self.label_bytes

    def set_label_map_path(self, path):
        if os.path.isfile(path):
            self.label_map_path = path


    def convert_real_image_to_tfrecord(self,path,export_dir=''):
        """
        Convert real image to tfrecord as specific input.
        Args:
            path: path to image file. 
                There should be 2 folders named: 'images' and 'boxes',
                'images' will have all image dataset and 'boxes' has all real detect box
                txt files map with the name in 'images' folder. Format is:
                [classification_id],[top_left_x],[top_left_y],[width],[height]
            export_dir: path to tfrecord file.            
        """
        self.path = path
        if export_dir == '':
            export_dir = path+"/../export/"

        # self._read_label_map()

        image_name_list = self._get_image_name_list()

        train_images,test_images = self._divide_train_test(image_name_list,lambda index: index >= 5,shuffle=True)

        train_export_dir = self._mkdir_if_not_exist(export_dir,"train")
        self._write_image_data_to_tfrecord(train_images,export_dir+"train/")

        test_export_dir = self._mkdir_if_not_exist(export_dir,"test")
        self._write_image_data_to_tfrecord(test_images,export_dir+"test/")

    def _read_label_map(self):
        self.label_map_dict = label_map_util.get_label_map_dict(self.label_map_path)

    def _read_tfrecord(self,path=""):
        path_list = [os.path.join(path,file) for file in os.listdir(path)]
        
        queue = tf.train.string_input_producer(path_list,num_epochs=None)
        
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'height':  tf.FixedLenFeature([],tf.int64),
                'width': tf.FixedLenFeature([],tf.int64),
                'depth': tf.FixedLenFeature([],tf.int64),
                'label': tf.FixedLenFeature([],tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string)
            }
        )

        ### Old ways
        # image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.image.decode_image(features['image_raw'],channels=3)

        label = tf.cast(features['label'], tf.int32)
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        depth = tf.cast(features['depth'], tf.int32)            
        # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        image = tf.reshape(image,shape=[height,width,3])

        return image, label

   

    def _get_image_name_list(self):
        
        image_name_list = os.listdir(os.path.join(self.path,"images"))
        
        return image_name_list

    def _divide_train_test(self,images,divide_logic,shuffle=False):
        train_images = []
        test_images = []

        if shuffle:
            self._shuffle(images);

        for i in range(len(images)):
            if divide_logic(i):
                test_images.append(images[i])
            else:
                train_images.append(images[i])

        return train_images,test_images

    def _shuffle(self,images):
        return random.shuffle(images)

    def _mkdir_if_not_exist(self,export_path,folder_name):
        new_path = os.path.join(export_path,folder_name)
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        return new_path

    def _write_image_data_to_tfrecord(self,image_name_list,export_dir,
                                        prefix_name="",max_image_in_file=1500,get_loop=True):
        """
        Write image bitmap pixel value array to tfrecord (space consuming)
        """
       
        
        num_image_in_file = 0
        record_name = 1
        writer = tf.python_io.TFRecordWriter(export_dir + prefix_name+"1.tfrecord")
        for i in range(len(image_name_list)):        

            num_image_in_file +=1
            image_name = image_name_list[i].split(".")[0]
            image_ext = image_name_list[i].split(".")[1]
            image_path = os.path.join(self.path,"images",image_name+"."+image_ext)
            
            image_array = self._convert_image_to_array(image_path)

            image_byte = self._convert_image_to_raw(image_path)
            
            bb_inside = self.get_bounding_boxes(image_name)

            features = {
                'height':  data_feature.int64_feature(len(image_array)),
                'width': data_feature.int64_feature(len(image_array[0])),
                'depth': data_feature.int64_feature(len(image_array[0][0])),
                'label': data_feature.int64_list_feature(bb_inside["label_int"]),
                'image_raw': data_feature.byte_feature(image_byte),
                'bbox/xmin': data_feature.int64_list_feature(bb_inside["bounding_box_xmin"]),
                'bbox/ymin': data_feature.int64_list_feature(bb_inside["bounding_box_ymin"]),
                'bbox/xmax': data_feature.int64_list_feature(bb_inside["bounding_box_xmax"]),
                'bbox/ymax': data_feature.int64_list_feature(bb_inside["bounding_box_ymax"]),
                
            }
            hidden_num = 0
            if "hidden" in bb_inside:
                for hidden in bb_inside["hidden"]:
                    features["hidden_"+str(hidden_num)] = data_feature.bytes_list_feature(hidden)
                    hidden_num +=1

            # Write to file
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())

            if ((num_image_in_file+1) % max_image_in_file == 0):
                writer.close()
                record_name += 1
                writer = tf.python_io.TFRecordWriter(export_dir+prefix_name+str(record_name)+".tfrecord")
                continue

        writer.close()


    def _convert_image_to_raw(self,image_path):            
        return tf.gfile.FastGFile(image_path,mode="rb").read()


    def _convert_image_to_array(self,image_path):
        image_data = cv2.imread(image_path)

        return image_data


    def train_data_tfrecord(self,train_path,batch_size,crop_size=None):
        
        image, label = self._read_tfrecord(train_path)
        kdm = KDataManipulating()
        if crop_size == None:
            crop_size = [64,64]
       
        image = kdm.crop(image,type="",crop_size=crop_size)
        image = kdm.standardize(image)


        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=100+3*batch_size,
            min_after_dequeue=100
            )
        
        labels = tf.reshape(sparse_labels,shape=[batch_size])
        
        return images, labels

    @abc.abstractmethod
    def get_bounding_boxes(self,image_path):
        raise NotImplementedError("""You have to implement this function which return array of object that contain:\n \
        bounding_box_xmin: array(bb_xmin),    
        bounding_box_ymin: array(bb_ymin),    
        bounding_box_xmax: array(bb_xmax),    
        bounding_box_ymax: array(bb_ymax),    
        label_int: array(label_int),          
        Added info, save anything you want  
        """)


class DataRecord(object):
    pass
        
