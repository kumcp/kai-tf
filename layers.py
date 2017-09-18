import tensorflow as tf
import re
    
class KLayer(object):
    
    @staticmethod
    def set_batchsize(batchsize):
        KLayer.BATCH_SIZE = batchsize
    
    @staticmethod
    def summary(x):
        TOWER_NAME = 'tower'
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    @staticmethod
    def conv_layer(inputs,shape,num_filter,initializer,bias_initializer=tf.constant_initializer(0.0),strides=[1,1,1,1],padding='SAME',wd=0.0,name='conv',reuse=None):
        
        with tf.variable_scope(name,reuse=reuse) as scope:
            initializer = tf.truncated_normal_initializer(stddev=1e-1)

            with tf.device('/cpu:0'):
                # shape=[5, 5, 3, conv1_node]
                kernel = tf.get_variable('weights', shape, initializer=initializer)
                
                weight_decay = tf.multiply(tf.nn.l2_loss(kernel), wd, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)
                biases = tf.get_variable('biases', shape=[num_filter],initializer=bias_initializer)

            conv = tf.nn.conv2d(inputs, kernel, strides, padding=padding)

            pre_activation = tf.nn.bias_add(conv, biases)
            conv_output = tf.nn.relu(pre_activation, name=scope.name)

            # Export to histogram.
            KLayer.summary(conv_output)
            return conv_output

    @staticmethod
    def fc_layer(inputs,shape,num_node,initializer,bias_initializer=tf.constant_initializer(0.1),wd=0.0,name='fc',reuse=None,reshape=False,relu=True):
        
        with tf.variable_scope(name,reuse=reuse) as scope:
            if reshape:
                reshape_inputs = tf.reshape(inputs, [KLayer.BATCH_SIZE, -1])
                dim = reshape_inputs.get_shape()[1].value
                shape=[dim, num_node]
            else:
                reshape_inputs = inputs
            
            with tf.device('/cpu:0'):
                
                weights = tf.get_variable('weights', shape, initializer=initializer)
            
                weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)
                biases = tf.get_variable('biases',[num_node],initializer=bias_initializer)

            
            if relu:
                pre = tf.add(tf.matmul(reshape_inputs, weights), biases)
                fc_output = tf.nn.relu(pre,name=scope.name)
            else:
                fc_output = tf.add(tf.matmul(reshape_inputs, weights), biases,name=scope.name)

            KLayer.summary(fc_output)

            return fc_output


    @staticmethod
    def batch_norm(inputs,is_training=True,name="batch_norm"):
        norm = tf.contrib.layers.batch_norm(inputs, center=True, scale=True, 
                                          is_training=is_training,
                                          scope=name+'_norm')
        return tf.nn.relu(norm,name=name+"_relu")

    @staticmethod
    def residual_layer(inputs,input_shape,output_shape,initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1),is_training=True,name="res"):
        

        first_shape = [1,1,input_shape[2], input_shape[3]]
        branch_a1 = KLayer.conv_layer(inputs,first_shape,input_shape[3],initializer,name=name+"_1a")
        # norm1 = tf.nn.batch_normalization(branch_a1, mean=0.0, variance=1.0, offset, beta=0.75,name=name+"norm_1a")
        norm_a1 = KLayer.batch_norm(branch_a1,is_training=is_training,name=name+"_1a_bn")

        input_shape[2] = input_shape[3]
        branch_a2 = KLayer.conv_layer(norm_a1,input_shape,input_shape[3],initializer,name=name+"_2a" )
        norm_a2 = KLayer.batch_norm(branch_a2,is_training=is_training,name=name+"_2a_bn")

        output_shape_a = output_shape[:]
        output_shape_a[2] = input_shape[3]
        branch_a3 = KLayer.conv_layer(norm_a2,output_shape_a,output_shape[3],initializer,name=name+"_3a" )
        norm_a3 = tf.contrib.layers.batch_norm(branch_a3, center=True, scale=True, 
                                          is_training=is_training,
                                          scope=name+'_3a_bn')
        
        branch_b1 = KLayer.conv_layer(inputs,output_shape,output_shape[3],initializer,name=name+"_3b" )
        norm_b1 = tf.contrib.layers.batch_norm(branch_b1, center=True, scale=True, 
                                          is_training=is_training,
                                          scope=name+'_1b_bn')

        return tf.add(norm_a3,norm_b1)