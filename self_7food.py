
import os.path
import tensorflow as tf
from classification import KClassification
from layers import KLayer

class K7FoodClassification(KClassification):

    def __init__(self):

        self.set_batch_size(32)
        self.log_dir = "./log"
               
        self.num_test = 30000
        self.force_start = False
        self.sess = tf.InteractiveSession()
        self.num_run = 10000
        self.ini_lr = 0.01
        self.decay_rate = 0.99
        self.decay_step = 100000
        self.crop_size=[80,80]
        self.model_name = "food_res_190917"
        self.eval_log_dir = "./log/"+self.model_name

    def set_batch_size(self,batch_size):
        self.BATCH_SIZE = batch_size
        KLayer.set_batchsize(batch_size)

    def model(self,images,reuse=None,is_training=True):
        TOWER_NAME = 'tower'

        conv1_node = 32
        conv1_shape = [5,5,3,conv1_node]
        
        conv2_node = 64
        conv2_shape = [5,5,conv1_node,conv2_node]

        conv3a_node = 64
        conv3_input_shape = [5,5,conv2_node,conv3a_node]

        conv3b_node = 128
        conv3_output_shape = [5,5,conv2_node,conv3b_node]
        

        local4_node = 100
        local5_node = 50

        initializer = tf.truncated_normal_initializer(mean=0.0,stddev=0.1)

        conv1 = KLayer.conv_layer(images,shape=conv1_shape,num_filter=conv1_node,initializer=initializer,
                                        name='conv1')
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1],padding='SAME', name='pool1')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')

        conv2 = KLayer.conv_layer(norm1,shape=conv2_shape,num_filter=conv2_node,initializer=initializer,
                                        name='conv2')
        pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1],padding='SAME', name='pool2')
        norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')

        

        normX3 = KLayer.residual_layer(norm2,conv3_input_shape,conv3_output_shape)

        fc3 = KLayer.fc_layer(normX3,shape=[local4_node],num_node=local4_node,initializer=initializer,
                                bias_initializer=initializer,wd=4e-5,
                                name='fc3',reshape=True)
        
        fc4 = KLayer.fc_layer(fc3,shape=[local4_node,local5_node],num_node=local5_node,initializer=initializer,
                                bias_initializer=initializer,wd=4e-4,
                                name='fc4')

        softmax_linear = KLayer.fc_layer(fc4,shape=[local5_node, self.NUM_CLASSES],num_node=self.NUM_CLASSES,
                                            initializer=tf.truncated_normal_initializer(stddev=1/192.0),
                                            bias_initializer=tf.constant_initializer(0.1),wd=4e-3,
                                            name='softmax_linear',
                                            relu=False)
       
        return softmax_linear
    