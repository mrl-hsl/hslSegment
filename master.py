import tensorflow as tf
import network as net
from data_provider import _parse_image_function, _flip_left_right, _crop_random, _one_hot_encode, _resize_data, _color
import config as cfg
import numpy as np
import os
import cv2

palette = cfg.color_palette
num_classes = len(palette)
input_width = cfg.INPUT_WIDTH
input_height = cfg.INPUT_HEIGHT


class Master():
    # def __init__(self, i,load,modelAddress):
    def __init__(self, log_index):
        self.log_index = log_index
        self.model = net.model
        self.input = net.net.input

        self.label = tf.placeholder(
            tf.float32, [None, input_height, input_width, num_classes], name='label')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.lrate = tf.placeholder(
            dtype=tf.float32, shape=[], name='learning_rate')
        # self.lrate = tf.train.piecewise_constant(
        #     self.global_step, [5000, 10000], [1e-3, 1e-4, 1e-5])
        tf.summary.scalar('lerning_rate', self.lrate)

        self.prepare_data()

        self.buildOptimizer()
        self.buildPredector()
        self.buildMetrics()

        # Initialize session(to avoid cuda internal error)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.train_writer = tf.summary.FileWriter(
            'logs/' + str(self.log_index) + "/train", self.sess.graph, flush_secs=10)
        # self.test_writer = tf.summary.FileWriter(
        #     'logs/' + str(self.log_index) + "/test", self.sess.graph, flush_secs=10)
        self.saver = tf.train.Saver()

        # if load == False:
        #     self.sess.run(tf.global_variables_initializer())
        #     self.sess.run(tf.local_variables_initializer())
        # else:
        #     self.saver.restore(self.sess, modelAddress)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(self.trainIt.initializer)
        # self.sess.run(self.testIt.initializer)
        self.graph = tf.get_default_graph()
        self.merged = tf.summary.merge_all()

        self.train_op_counter = 0

    def prepare_data(self):
        dataset = tf.data.TFRecordDataset(
            ['./tfrecords/train.tfrecords']
        )
        dataset = dataset.map(_parse_image_function)
        dataset = dataset.map(_flip_left_right)
        dataset = dataset.map(_crop_random)
        dataset = dataset.map(_one_hot_encode)
        dataset = dataset.map(_resize_data)
        dataset = dataset.map(_color)
        dataset = dataset.shuffle(buffer_size=50)
        train_set = dataset
        # train_set = train_set.batch(16)
        train_set = train_set.repeat(100)
        train_iterator = train_set.make_initializable_iterator()
        next_train_batch = train_iterator.get_next()

        self.trainBatch = next_train_batch
        self.trainIt = train_iterator

        # self.testBatch = data.next_test_batch
        # self.testIt = data.test_iterator

    def buildOptimizer(self):
        with tf.name_scope('optimizer'):
            weights = self.label * np.ones(num_classes)
            weights = tf.reduce_sum(weights, 3)
            self.loss = tf.losses.softmax_cross_entropy(
                onehot_labels=self.label, logits=self.model, weights=weights)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.lrate, name='adam_optimizer')
                self.train_op = self.optimizer.minimize(
                    self.loss, global_step=self.global_step, name='train_op')
            tf.summary.scalar('total_loss', self.loss)

    def buildPredector(self):
        with tf.name_scope('predictor'):
            self.softmax_output = tf.nn.softmax(
                self.model, name='softmax_output')
            self.predictions_argmax = tf.argmax(
                self.softmax_output, axis=-1, name='predictions_argmax', output_type=tf.int64)

    def buildMetrics(self):
        with tf.variable_scope('metrics') as scope:
            self.labels_argmax = tf.argmax(
                self.label, axis=-1, name='labels_argmax', output_type=tf.int64)
            self.acc_value, self.acc_update_op = tf.metrics.accuracy(
                labels=self.labels_argmax, predictions=self.predictions_argmax)
            self.acc_value = tf.identity(self.acc_value, name='acc_value')
            self.acc_update_op = tf.identity(
                self.acc_update_op, name='acc_update_op')
            self.local_metric_vars = tf.contrib.framework.get_variables(
                scope=scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
            self.metrics_reset_op = tf.variables_initializer(
                var_list=self.local_metric_vars, name='metrics_reset_op')
            self.accuracy = tf.summary.scalar('accuracy', self.acc_value)

            self.accPerClass, self.accPerClassOp = tf.metrics.mean_per_class_accuracy(
                self.labels_argmax, self.predictions_argmax, num_classes)

            for class_name, index in zip(palette, range(len(palette))):
                tf.summary.scalar(class_name, self.accPerClassOp[index])

    def saveModel(self):
        path = './models/' + str(self.train_op_counter)
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
            # print("The new directory is created!")
        self.saver.save(self.sess, path + '/model')

    def teach(self):
        while True:
            try:
                b = self.sess.run(self.trainBatch)
                image = b[0]
                # show_image(image)
                image = image.reshape((1, input_height, input_width, 3))
                label = b[1]
                # show_label(label)
                label = label.reshape(
                    (1, input_height, input_width, num_classes))
                summary, opt = self.sess.run([self.merged, self.train_op,], feed_dict={
                                             self.input: image, self.label: label, self.lrate: 0.001})
                # self.sess.run(self.metrics_reset_op)
                self.train_writer.add_summary(summary, self.train_op_counter)
                summary, valLose, op_update, op_PerClass = self.sess.run([self.merged, self.loss, self.acc_update_op, self.accPerClassOp], feed_dict={
                                                                         self.input: image, self.label: label, self.lrate: 0.001})
                _acc_value = self.sess.run(self.acc_value)
                accperclass = self.sess.run(self.accPerClass)
                print(self.train_op_counter, "valLoss, ",
                      valLose, _acc_value, op_PerClass)

                if _acc_value > 0.96 or self.train_op_counter % 500 == 0:
                    self.saveModel()

                self.train_op_counter += 1
            except tf.errors.OutOfRangeError:
                self.sess.run(self.trainIt.initializer)


master = Master(3)
master.teach()
