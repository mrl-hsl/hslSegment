import tensorflow as tf

class Network():
    def __init__(self):
        self.input = tf.placeholder(tf.float32, [None, 240, 320, 3], name='input')
        self.isTraining = tf.placeholder(tf.bool, [], name='isTraining')
        
        self.activation = tf.nn.relu6
        self.weightInitializer = tf.contrib.layers.xavier_initializer
        self.normalizer_fn = None
        self.norm_params = None

    def separableConvMobileNet(self, features, kernel_size, out_filters, stride, _name, dilationFactor=1, pad='SAME'):
        with tf.variable_scope(_name):
            output = tf.layers.separable_conv2d(
                features,
                depth_multiplier=1,
                kernel_size=kernel_size,
                strides=stride,
                padding=pad,
                activation=self.activation,
                depthwise_initializer=self.weightInitializer(),
                pointwise_initializer=self.weightInitializer(),
                depthwise_regularizer=None,
                pointwise_regularizer=None,
                use_bias=False,
            )

            output = tf.layers.conv2d(
                output,
                filters=out_filters,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='SAME',
                activation=self.activation,
                kernel_initializer=self.weightInitializer(),
                kernel_regularizer=None,
                use_bias=False,
            )

            return output

    def model(self):
        conv1 = self.separableConvMobileNet(self.input, 3, 8, 2, 'conv1')
        
        conv2 = self.separableConvMobileNet(conv1, 3, 16, 1, 'conv2')
        conv2 = tf.layers.average_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='VALID')
        
        conv3 = self.separableConvMobileNet(conv2, 3, 32, 1, 'conv3')
        conv3 = tf.layers.average_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='VALID')

        conv4 = self.separableConvMobileNet(conv3, 3, 32, 1, 'conv4')
        conv4 = tf.layers.average_pooling2d(conv4, pool_size=(2, 2), strides=(2, 2), padding='VALID')

        conv5 = self.separableConvMobileNet(conv4, 3, 32, 1, 'conv5', dilationFactor=2)
        conv6 = self.separableConvMobileNet(conv5, 3, 32, 1, 'conv6', dilationFactor=4)

        o1 = tf.image.resize_bilinear(conv6, [60, 80])

        conv7 = self.separableConvMobileNet(o1, 3, 64, 1, 'conv7')

        o2 = tf.image.resize_bilinear(conv7, [240, 320])

        conv8 = self.separableConvMobileNet(o2, 3, 64, 1, 'conv8')
        
        conv9 = self.separableConvMobileNet(conv8, 3, 3, 1, 'conv9')
        
        return conv9

net = Network()
model = net.model()
model = tf.identity(model, name="model")

if __name__ == '__main__':
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs/graphs', sess.graph)