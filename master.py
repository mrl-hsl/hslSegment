import tensorflow as tf
import network as net
from data_provider import _parse_image_function, _flip_left_right, _crop_random, _one_hot_encode, _resize_data, _color, tfrecord_data_image_to_opencv_mat, cv_show_image, one_hot_image_matrix_to_label, label_matrix_to_label
import config as cfg
import numpy as np
import os
import cv2

palette = cfg.color_palette
num_classes = len(palette)
input_width = cfg.INPUT_WIDTH
input_height = cfg.INPUT_HEIGHT
max_models_to_keep = cfg.MAX_MODELS_TO_KEEP
debug_enabled = cfg.DEBUG_ENABLED


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
        self.test_writer = tf.summary.FileWriter(
            'logs/' + str(self.log_index) + "/test", self.sess.graph, flush_secs=10)
        self.saver = tf.train.Saver(max_to_keep=max_models_to_keep)

        # if load == False:
        #     self.sess.run(tf.global_variables_initializer())
        #     self.sess.run(tf.local_variables_initializer())
        # else:
        #     self.saver.restore(self.sess, modelAddress)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(self.trainIt.initializer)
        self.sess.run(self.testIt.initializer)
        self.graph = tf.get_default_graph()
        self.merged = tf.summary.merge_all()

        self.train_op_counter = 0

    def prepare_data(self):
        train_dataset = tf.data.TFRecordDataset(
            ['./train.tfrecords']
        )
        train_dataset = train_dataset.map(_parse_image_function)
        train_dataset = train_dataset.map(_flip_left_right)
        train_dataset = train_dataset.map(_crop_random)
        train_dataset = train_dataset.map(_one_hot_encode)
        train_dataset = train_dataset.map(_resize_data)
        train_dataset = train_dataset.map(_color)
        train_dataset = train_dataset.shuffle(buffer_size=50)
        # train_set = train_set.batch(16)
        # train_set = train_set.repeat(100)
        train_iterator = train_dataset.make_initializable_iterator()
        next_train_batch = train_iterator.get_next()
        self.trainBatch = next_train_batch
        self.trainIt = train_iterator

        test_dataset = tf.data.TFRecordDataset(
            ['./test.tfrecords']
        )
        test_dataset = test_dataset.map(_parse_image_function)
        # test_dataset = test_dataset.map(_flip_left_right)
        # test_dataset = test_dataset.map(_crop_random)
        test_dataset = test_dataset.map(_one_hot_encode)
        test_dataset = test_dataset.map(_resize_data)
        # test_dataset = test_dataset.map(_color)
        # test_dataset = test_dataset.shuffle(buffer_size=50)
        test_dataset  = test_dataset.batch(1)
        test_iterator = test_dataset.make_initializable_iterator()
        next_test_batch = test_iterator.get_next()
        self.testBatch = next_test_batch
        self.testIt = test_iterator

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
        path_exists = os.path.exists(path)
        if not path_exists:
            os.makedirs(path)

        model_meta_path = path + '/meta'
        model_meta_path_exists = os.path.exists(model_meta_path)
        if not model_meta_path_exists:
            os.makedirs(model_meta_path)

        model_pb_path = path + '/pb'
        model_pb_path_exists = os.path.exists(model_pb_path)
        if not model_pb_path_exists:
            os.makedirs(model_pb_path)

        self.saver.save(self.sess, model_meta_path + '/saved_model')
        tf.saved_model.simple_save(self.sess, model_pb_path, inputs={
                                   "input": self.input}, outputs={"model": self.softmax_output})

    def eval(self):
        try:
            batch = self.sess.run(self.testBatch)
            feed = {
                self.input: batch[:][0], self.label: batch[:][1], self.lrate: 0.001}
            summary, valLose, predictions = self.sess.run(
                [self.merged, self.loss, self.predictions_argmax], feed_dict=feed)
            self.test_writer.add_summary(summary, self.train_op_counter)
            # print(self.train_op_counter, "**************, ", valLose)
            # print("predictions:", predictions.shape)
            for i in range(batch[0].shape[0]):
                seg = predictions[i]
                print(batch[:][0].shape)
                cv_image = tfrecord_data_image_to_opencv_mat(batch[:][0][0])
                cv_show_image(cv_image, "image", 1)
                cv_ground_truth = one_hot_image_matrix_to_label(batch[:][1][0])
                cv_show_image(cv_ground_truth, "ground_truth", 1)
                cv_label = label_matrix_to_label(seg)
                cv_show_image(cv_label, "label", 0)
        except tf.errors.OutOfRangeError:
            self.sess.run(self.testIt.initializer)

    def teach(self):
        while True:
            try:
                b = self.sess.run(self.trainBatch)
                image = b[0]
                # cv_image = tfrecord_data_image_to_opencv_mat(image)
                # cv_show_image(cv_image, "image", 1)
                image = image.reshape((1, input_height, input_width, 3))
                label = b[1]
                # cv_label = one_hot_image_matrix_to_label(label)
                # cv_show_image(cv_label, "label", 0)
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

                if (debug_enabled):
                    if self.train_op_counter % 500 == 0:
                        self.eval()

                self.train_op_counter += 1
            except tf.errors.OutOfRangeError:
                self.sess.run(self.trainIt.initializer)


master = Master(3)
master.teach()
