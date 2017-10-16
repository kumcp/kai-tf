from readerwriter import KReaderWriter
import tensorflow as tf
import numpy as np
import os.path

import math
import time
from datetime import datetime


class KClassification(object):
    NUM_CLASSES = 7

    def __init__(self):
        self.log_dir = "./log"
        self.data_dir = os.path.join(os.path.realpath(__file__), '../cifar10')
        self.eval_log_dir = "./log/eval"
        self.sess = tf.InteractiveSession()
        self.force_start = True
        self.num_run = 2000
        self.num_test = 10000
        self.decay_step = 1000

    def read_data(self, data_dir):
        krw = KReaderWriter()
        return krw.train_data_tfrecord(data_dir, batch_size=self.BATCH_SIZE, crop_size=self.crop_size)

    def train(self, train_dir=None):
        if train_dir == None:
            train_dir = self.data_dir

        images, labels = self.read_data(train_dir)
        tf.summary.image('images', images)
        logits = self.model(images)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        tf.summary.scalar('cross_entropy', cross_entropy[0])
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        loss = total_loss

        global_step = tf.contrib.framework.get_or_create_global_step()

        ini_lr = self.ini_lr
        decay_rate = self.decay_rate
        decay_steps = self.decay_step

        lr = tf.train.exponential_decay(
            ini_lr, global_step, decay_steps, decay_rate, staircase=True)
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
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        # Create the graph, etc.

        # Create a session for running operations in the Graph.
        # sess = tf.Session()

        # sess = tf.train.MonitoredTrainingSession(
        #     checkpoint_dir=self.log_dir,
        #     hooks=[tf.train.StopAtStepHook(num_steps=100000),
        #         tf.train.NanTensorHook(init_op),
        #         _LoggerHook()],
        #     config=tf.ConfigProto(log_device_placement=False)
        #     )

        # Initialize the variables (like the epoch counter).
        self.restore_session(force_start=self.force_start)
        # sess.run(tf.local_variables_initializer())

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        num_run = 0

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.eval_log_dir)

        try:
            while not coord.should_stop():
                num_run += 1
                if num_run % 10 == 0:
                    _, loss_value, sum_up = self.sess.run(
                        [train_op, loss, summary_op])
                    self.prev_num += 1
                    print("Step: %d , Loss: %f" % (self.prev_num, loss_value))
                    summary_writer.add_summary(sum_up, self.prev_num)

                else:
                    self.sess.run([train_op])
                if num_run > self.num_run:
                    break

        except tf.errors.OutOfRangeError:
            print('Done training!!!')

        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        self.save_session()
        self.sess.close()

    def classify(self, image_path):
        image_array = cv2.imread(image_path)

        height = image_array.shape[0]
        width = image_array.shape[1]

        image_data = tf.placeholder(tf.int32, shape=[height, width, 3])

        kdm = KDataManipulating()

        image = kdm.crop(image_data, type="resize", crop_size=self.crop_size)
        image = tf.cast(kdm.standardize(image), tf.float32)

        with tf.Session() as tempsess:
            image_value = tempsess.run(image, {image_data: image_array})

        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(self.sess, coord=coord, daemon=True,
                                                 start=True))

            softmax_value = self.sess.run(tf.contrib.layers.softmax(
                self.logit), feed_dict={self.image_input: image_value})

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

        labels = self.labels

        result = dict(zip(labels, softmax_value[0]))

        return result

    def prepare_classify(self):

        self.image_input = tf.placeholder(
            tf.float32, shape=[self.crop_size[0], self.crop_size[1], 3])
        # images = tf.train.batch([self.image_input],batch_size=1)
        images = tf.expand_dims(self.image_input, 0)
        self.BATCH_SIZE = 1
        self.logit = self.model(images, is_training=False)
        self.restore_session()

    def model(self, images, reuse=None, is_training=True):
        raise NotImplementedError(
            "You have to define your own model in your inherited class")

    def save_session(self, path=''):
        saver = tf.train.Saver()
        if path == '':
            path = os.path.dirname(os.path.realpath(
                __file__)) + "/result/" + self.model_name + "/model.test"
        saver.save(self.sess, path)
        print("Model saved to: %s" % path)
        with open('num_run.txt', 'w') as myfile:
            myfile.seek(0)
            myfile.truncate()
            myfile.write(str(self.prev_num))
        print("Save from step %s" % self.prev_num)
        return self.sess

    def restore_session(self, path='', force_start=False):
        saver = tf.train.Saver()

        if path == '':
            path = os.path.dirname(os.path.realpath(
                __file__)) + "/result/" + self.model_name + "/model.test"
        if os.path.isfile(path + ".index") and not force_start:
            saver.restore(self.sess, path)
            print("Loaded model from: %s" % path)
            with open('num_run.txt', 'r') as myfile:
                self.prev_num = int(myfile.read().replace('\n', ''))
                print("Continue run from step %s" % self.prev_num)
        else:
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            self.prev_num = 0
            if tf.gfile.Exists(self.eval_log_dir):
                tf.gfile.DeleteRecursively(self.eval_log_dir)
            tf.gfile.MakeDirs(self.eval_log_dir)
            print('Run from the start')

    def eval(self, eval_dir=None, train_dir=None, eval_interval_sec=60 * 5, once=True):

        self.coord = tf.train.Coordinator()

        if eval_dir == None:
            eval_dir = self.data_dir

        images, labels = self.read_data(eval_dir)
        logits = self.model(images, is_training=False)

        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        MOVING_AVERAGE_DECAY = 0.9999
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(self.eval_log_dir)

        while True:
            self.eval_once(saver, summary_writer, top_k_op, summary_op)
            if once:
                break

    def eval_once(self, saver, summary_writer, top_k_op, summary_op, restore=True):

        if restore:
            self.restore_session()

        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(self.sess, coord=self.coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(self.num_test / self.BATCH_SIZE))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * self.BATCH_SIZE
            step_iter = 0
            while step_iter < num_iter and not self.coord.should_stop():
                predictions = self.sess.run([top_k_op])
                true_count += np.sum(predictions)
                step_iter += 1

            # Compute precision @ 1.
            precision = float(true_count) / total_sample_count * 100
            print('%s: precision @ 1 = %.3f %%' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(self.sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)

            # Add summary as continue from next_step
            # summary_writer.add_summary(summary, next_step)
        except Exception as e:  # pylint: disable=broad-except
            self.coord.request_stop(e)

        self.coord.request_stop()
        self.coord.join(threads, stop_grace_period_secs=10)


class _LoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime."""

    log_frequency = 10      # Log each 10 batch-run

    def __init__(self, loss, lr, batch_size):
        self.loss = loss
        self.lr = lr
        self.batch_size = batch_size

    def begin(self):
        self._step = -1
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        # Asks for loss value.
        return tf.train.SessionRunArgs([self.loss, self.lr])

    def after_run(self, run_context, run_values):
        if self._step % self.log_frequency == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time

            loss_value = run_values.results[0]
            examples_per_sec = self.log_frequency * self.batch_size / duration
            sec_per_batch = float(duration / self.log_frequency)

            learning_rate_value = run_values.results[1]
            format_str = ('%s: step %d, loss = %.2f, lr= %.6f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print (format_str % (datetime.now(), self._step, loss_value, learning_rate_value,
                                 examples_per_sec, sec_per_batch))
