import tensorflow as tf
import cv2
import os
import random
from datamanipulating import KDataManipulating

class KReaderWriter(object):
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

    def read_from_binary(self,queue):

        #Read each file as queue
        filename_queue = tf.train.string_input_producer(queue)
        
        result = DataRecord()  
        reader = tf.FixedLengthRecordReader(record_bytes=self.record_bytes)

        #Read as key value, value is a string
        result.key, value = reader.read(filename_queue)

        # Decode string into uint8
        record_bytes = tf.decode_raw(value, tf.uint8)

        result.label = tf.cast(tf.strided_slice(record_bytes, [0], [self.label_bytes]), tf.int32)

        depth_major = tf.reshape(tf.strided_slice(record_bytes, [self.label_bytes],
                       [self.record_bytes]),
                       [self.depth, self.height, self.width])

        result.uint8image = tf.transpose(depth_major, [1, 2, 0])

        return result

    def convert_real_image_to_tfrecord(self,path,export_dir='',get_loop=True):
        self.path = path
        if export_dir == '':
            export_dir = path+"/../export/"

        image_name_list,image_sum = self._get_image_name_list()
        
        train_sum, train_images,test_sum,test_images = self._divide_train_test(image_name_list,lambda index: index >= 800,True)

        train_export_dir = self._mkdir_if_not_exist(export_dir,"train")
        self._write_image_data_to_tfrecord(train_sum,train_images,export_dir+"train/")

        test_export_dir = self._mkdir_if_not_exist(export_dir,"test")
        self._write_image_data_to_tfrecord(test_sum,test_images,export_dir+"test/")

        return ''

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

    def _int64_feature(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _byte_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _get_image_name_list(self):
        
        self.labels = os.listdir(self.path)
        LABEL_NUMS = len(self.labels)
        
        image_name_list = []
        image_sum = []

        for label in self.labels:
            image_name_list_in_class = os.listdir(os.path.join(self.path,label))
            image_name_list.append(image_name_list_in_class)
            image_sum.append(len(image_name_list_in_class))
            

        return image_name_list, image_sum

    def _divide_train_test(self,images,divide_logic,shuffle=False):
        train_sum = []
        train_images = []
        test_sum = [] 
        test_images = []

        if shuffle:
            self._shuffle(images);

        for list_image in images:
            train_row = []
            test_row = []
            
            for i in range(len(list_image)):
                if divide_logic(i):
                    test_row.append(list_image[i])
                else:
                    train_row.append(list_image[i])

            train_images.append(train_row)
            train_sum.append(len(train_row))
            test_images.append(test_row)
            test_sum.append(len(test_row))

        return train_sum, train_images, test_sum, test_images

    def _shuffle(self,images):
        for image_label in images:
            random.shuffle(image_label)
        return images

    def _mkdir_if_not_exist(self,export_path,folder_name):
        new_path = os.path.join(export_path,folder_name)
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        return new_path

    def _write_image_data_to_tfrecord(self,sum_image,image_name_list,export_dir,
                                        prefix_name="",max_image_in_file=1500,get_loop=True):
        """
        Write image bitmap pixel value array to tfrecord (space consuming)
        """
        LABEL_NUMS = len(self.labels)
        max_possible_image = LABEL_NUMS*max(sum_image)
        num_image_in_file = 0
        record_name = 1
        writer = tf.python_io.TFRecordWriter(export_dir + prefix_name+"1.tfrecord")
        for i in range(max_possible_image):
            label_int = int(i % LABEL_NUMS)
            image_int = int(i/LABEL_NUMS)
            if image_int >= len(image_name_list[label_int]):
                if get_loop:
                    image_int = image_int % len(image_name_list[label_int])
                else:
                    continue


            num_image_in_file +=1
            image_name = image_name_list[label_int][image_int]
            image_path = os.path.join(self.path,self.labels[label_int],image_name)
            print(image_path)
            
            image_array = self._convert_image_to_array(image_path)

            ### Old ways
            # image_byte = image_array

            image_byte = self._convert_image_to_raw(image_path)
            

            # Write to file
            example = tf.train.Example(features=tf.train.Features(feature={
                'height':  self._int64_feature(len(image_array)),
                'width': self._int64_feature(len(image_array[0])),
                'depth': self._int64_feature(len(image_array[0][0])),
                'label': self._int64_feature(label_int),
                'image_raw': self._byte_feature(image_byte)
            }))
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
        image_data.tostring()
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

class DataRecord(object):
    pass
        
