import os.path
import math
import tensorflow as tf
import numpy as np
import re
import time
from datetime import datetime

# from self_define import MyClassification
from readerwriter import KReaderWriter
from datamanipulating import KDataManipulating as kdm
from classification import KClassification, _LoggerHook

class KCifarClassification(KClassification):

    def __init__(self):
        self.log_dir = "./log"
        self.data_dir = os.path.join(os.path.realpath(__file__),'../cifar10')
        self.eval_log_dir = "./log/eval"
        self.BATCH_SIZE = 128
        self.num_examples = 10000

    def train(self):
        # Load train data:
        filenames = [os.path.join(self.data_dir,'data_batch_%d.bin' % i) for i in range(1,6)]

        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
        batch_size = self.BATCH_SIZE

        images, labels = self.read_data(filenames,NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
        
        logits = self.model(images)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        loss = total_loss

        global_step = tf.contrib.framework.get_or_create_global_step()


        """
        Number of images per epoch = 50000
        Number of images per batch = 128
        Number of batches per epoch = 50000/128

        Number or epoch per decay = 350
        -> Number of batches per decay = 350 * 50000/128

        new decay lr = old lr * decay_rate
        """
        ini_lr = 0.1
        decay_rate = 0.95
        
        lr = tf.train.exponential_decay(ini_lr, global_step, 1000,decay_rate,staircase=True)
        tf.summary.scalar('learning_rate', lr)

        # Loss summary
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.summary.scalar(l.op.name + ' (raw)', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.GradientDescentOptimizer(lr)
            grads = opt.compute_gradients(total_loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        MOVING_AVERAGE_DECAY = 0.9999
        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')


        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=self.log_dir,
            hooks=[tf.train.StopAtStepHook(num_steps=1000),
                tf.train.NanTensorHook(total_loss),
                _LoggerHook(loss,lr)],
            config=tf.ConfigProto(log_device_placement=False)
            ) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run([train_op,grads])
                

        return ''


    def classify(self):
        pass

    def model(self,images):
        TOWER_NAME = 'tower'
        
        with tf.variable_scope('conv1') as scope:
            initializer = tf.truncated_normal_initializer(stddev=5e-2)

            with tf.device('/cpu:0'):
                shape=[5, 5, 3, 64]
                kernel = tf.get_variable('weights', shape, initializer=initializer)
                wd=0.0
            
            # if wd is not None:
                weight_decay = tf.multiply(tf.nn.l2_loss(kernel), wd, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)
            conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding='SAME')

            with tf.device('/cpu:0'):
                biases = tf.get_variable('biases', shape=[64],initializer=tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)

            # Export to histogram.
            self.summary(conv1)
        
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1],
                                padding='SAME', name='pool1')

        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')


        with tf.variable_scope('conv2') as scope:
            initializer = tf.truncated_normal_initializer(stddev=5e-2)

            with tf.device('/cpu:0'):
                shape=[5, 5, 64, 64]
                kernel = tf.get_variable('weights', shape, initializer=initializer)
                wd = 0.0
            
            # if wd is not None:
                weight_decay = tf.multiply(tf.nn.l2_loss(kernel), wd, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)

            conv = tf.nn.conv2d(norm1, kernel, [1,1,1,1], padding='SAME')

            with tf.device('/cpu:0'):
                biases = tf.get_variable('biases', shape=[64],initializer=tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)

            # Export to histogram.
            self.summary(conv2)
        
        
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1],
                                padding='SAME', name='pool1')
                    
        with tf.variable_scope('local3') as scope:
            #Reshape
            reshape = tf.reshape(pool2, [128, -1])
            dim = reshape.get_shape()[1].value
            
            initializer = tf.truncated_normal_initializer(stddev=0.04)
            with tf.device('/cpu:0'):
                shape=[dim, 384]
                weights = tf.get_variable('weights', shape, initializer=initializer)
                
                wd = 0.004
                # if wd is not None:
                weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)

                biases = tf.get_variable('biases',[384],initializer=tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases,name=scope.name)
            self.summary(local3)

        with tf.variable_scope('local4') as scope:
            
            initializer = tf.truncated_normal_initializer(stddev=0.04)
            with tf.device('/cpu:0'):
                shape=[384, 192]
                weights = tf.get_variable('weights', shape, initializer=initializer)

                wd = 0.004
                # if wd is not None:
                weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)

                biases = tf.get_variable('biases',[192],initializer=tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases,name=scope.name)
            self.summary(local4)

        with tf.variable_scope('softmax_linear') as scope:
            initializer = tf.truncated_normal_initializer(stddev=1/192.0)
            with tf.device('/cpu:0'):
                shape=[192, self.NUM_CLASSES]
                weights = tf.get_variable('weights', shape, initializer=initializer)

                wd = 0.0
                # if wd is not None:
                weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)

                biases = tf.get_variable('biases',[self.NUM_CLASSES],initializer=tf.constant_initializer(0.1))
            
            softmax_linear = tf.add(tf.matmul(local4,weights), biases, name=scope.name)

            self.summary(softmax_linear)
        
        return softmax_linear

    NUM_CLASSES = 10
        
    def read_data(self,filenames,num_examples_per_epoch):
        height = 32
        width = 32
        depth = 3
        label_bytes = 1 #(10 labels < 1byte ~ 255 labels)

        data_size = [height, width, depth]
        kreader = KReaderWriter()
        kreader.set_size(data_size, label_bytes)
        result = kreader.read_from_binary(filenames)
        
        # Change some input randomly for train
        height_crop = 24
        width_crop = 24 # Input size for train
        NUM_CLASSES = 10
        
        batch_size=self.BATCH_SIZE

        reshaped_image = kdm.cast(result.uint8image, tf.float32)
        distorted_image = kdm.crop(reshaped_image,type="random",shape=[height_crop, width_crop, 3])
        distorted_image = kdm.flip(distorted_image)
        float_image = kdm.standardize(distorted_image)
        
        # Set shape
        float_image.set_shape([height_crop, width_crop, 3])
        result.label.set_shape([1])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                                min_fraction_of_examples_in_queue)

        num_preprocess_threads = 16
        # if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [float_image, result.label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
        # else:
        #     images, label_batch = tf.train.batch(
        #         [image, label],
        #         batch_size=batch_size,
        #         num_threads=num_preprocess_threads,
        #         capacity=min_queue_examples + 3 * batch_size)

        tf.summary.image('images', images)

        labels = tf.reshape(label_batch, [batch_size])
        labels = tf.cast(labels, tf.int64)

        return images, labels
        

    def eval(self,eval_interval_secs=60*5, once=False):
        with tf.Graph().as_default() as g:
            # Get images and labels for CIFAR-10.
            filenames = [os.path.join(self.data_dir, 'test_batch.bin')]

            NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
            MOVING_AVERAGE_DECAY = 0.9999

            images, labels = self.read_data(filenames,NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)

            # Build a Graph that computes the logits predictions from the
            # inference model.
            logits = self.model(images)
            
            # Calculate predictions.
            top_k_op = tf.nn.in_top_k(logits, labels, 1)

            # Restore the moving average version of the learned variables for eval.
            variable_averages = tf.train.ExponentialMovingAverage(
                MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()

            summary_writer = tf.summary.FileWriter(self.eval_log_dir, g)

            while True:
                self.eval_once(saver, summary_writer, top_k_op, summary_op)
                if once:
                    break
                # time.sleep(eval_interval_secs)


    def eval_once(self,saver, summary_writer, top_k_op, summary_op):
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.log_dir)
            
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                
            else:
                print('No checkpoint file found')
                return

            # Start the queue runners.
            coord = tf.train.Coordinator()
            
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                    start=True))

                num_iter = int(math.ceil(self.num_examples / self.BATCH_SIZE))
                true_count = 0  # Counts the number of correct predictions.
                total_sample_count = num_iter * self.BATCH_SIZE
                step = 0
                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    step += 1
                
                # Compute precision @ 1.
                precision = float(true_count) / total_sample_count * 100
                print('%s: precision @ 1 = %.3f %%' % (datetime.now(), precision))
                
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Precision @ 1', simple_value=precision)
                summary_writer.add_summary(summary, global_step)
            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

    