import tensorflow as tf
import network as net
import dataProvider as data

class Master():
    def __init__(self, i, load, modelAddress):
        self.model = net.model
        self.input = net.net.input
        self.labels = tf.placeholder(tf.float32, [None, 240, 320, 3], name='labels')
        
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.lrate = tf.train.piecewise_constant(self.global_step, [5000, 10000], [1e-3, 1e-4, 1e-5])
        tf.summary.scalar('learning_rate', self.lrate)

        self.trainBatch = data.next_train_batch
        self.trainIt = data.train_iterator

        self.testBatch = data.next_test_batch
        self.testIt = data.test_iterator

        self.buildOptimizer()
        self.buildPredector()
        self.buildMetrics()

        self.sess = tf.Session()

        self.train_writer = tf.summary.FileWriter('logs/' + str(i) + "/train", self.sess.graph, flush_secs=10)
        self.test_writer = tf.summary.FileWriter('logs/' + str(i) + "/test", self.sess.graph, flush_secs=10)
        self.saver = tf.train.Saver()
        self.sess.run(self.trainIt.initializer)
        self.sess.run(self.testIt.initializer)
        if load == False:
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
        else:
            self.saver.restore(self.sess, modelAddress)
        
        self.graph = tf.get_default_graph()
        self.merged = tf.summary.merge_all()

    def buildOptimizer(self):
        with tf.name_scope('optimizer'): 
            weights = self.labels * [1.0, 1.0, 1.0]
            weights = tf.reduce_sum(weights, 3)
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.labels, logits=self.model, weights=weights)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            loss_scale = 1e4
            loss_scaled = self.loss * loss_scale

            optimizer = tf.train.AdamOptimizer(learning_rate=self.lrate, name='adam_optimizer')

            def clip_grads(grads_and_vars):
                gradients, variables = zip(*grads_and_vars)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
                return list(zip(clipped_gradients, variables))

            mixed_precision_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer,
                                                                                                      loss_scale=loss_scale)

            grads_and_vars = mixed_precision_optimizer.compute_gradients(loss_scaled)
            clipped_grads_and_vars = clip_grads(grads_and_vars)
            self.train_op = mixed_precision_optimizer.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)

            tf.summary.scalar('total_loss', self.loss)

    def buildPredector(self):
        with tf.name_scope('predictor'):
            self.softmax_output = tf.nn.softmax(self.model, name='softmax_output')
            self.predictions_argmax = tf.argmax(self.softmax_output, axis=-1, name='predictions_argmax', output_type=tf.int64)
