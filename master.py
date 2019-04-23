import tensorflow as tf
import network as net
import dataProvider as data

class master():
    def __init__(self, i,load,modelAddress):
        self.model = net.model
        self.input = net.net.input
        self.labels = tf.placeholder(tf.float32, [None, 240, 320,3], name='labels')
        
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.lrate = tf.train.piecewise_constant(self.global_step, [5000,10000],[1e-3,1e-4,1e-5])
        tf.summary.scalar('lerning_rate',self.lrate)

        self.trainBatch = data.next_train_batch
        self.trainIt = data.train_iterator

        self.testBatch = data.next_test_batch
        self.testIt = data.test_iterator

        self.buildOptimizer()
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # self.train_op = tf.group([self.train_op, update_ops])
        
        self.buildPredector()
        self.buildMetrics()

        self.sess = tf.Session()

        self.train_writer = tf.summary.FileWriter('logs/' + str(i) + "/train", self.sess.graph, flush_secs=10)
        self.test_writer = tf.summary.FileWriter('logs/' + str(i) + "/test", self.sess.graph, flush_secs=10)
        self.saver = tf.train.Saver()
        self.sess.run(self.trainIt.initializer)
        self.sess.run(self.testIt.initializer)
        if load==False:
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
        else:
            self.saver.restore(self.sess, modelAddress)
        
        self.graph = tf.get_default_graph()
        self.merged = tf.summary.merge_all()
