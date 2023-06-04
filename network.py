import tensorflow as tf


class network():
    def __init__(self):
        self.input = tf.placeholder(
            tf.float32, [None, 240, 320, 3], name='input')
        self.isTraning = tf.placeholder(tf.bool, [], name='isTraning')

        self.activation = tf.nn.relu6
        self.weightInitializer = tf.contrib.layers.xavier_initializer
        self.normalizer_fn = None
        # tf.contrib.layers.batch_norm
        self.norm_params = None
        # {"is_training": self.isTraning}

    def conv2d(self, input, nFilters, kernelSize, _strides, _name):
        output = tf.contrib.layers.conv2d(
            inputs=input,
            num_outputs=nFilters,
            kernel_size=kernelSize,
            stride=_strides,
            scope=_name,
            weights_initializer=self.weightInitializer(),
            normalizer_fn=self.normalizer_fn,
            normalizer_params=self.norm_params,
            activation_fn=self.activation,
            weights_regularizer=None,
            padding="SAME"
        )
        return output

    def maxPooling(self, inputs, kernelSize, strides):
        return tf.contrib.layers.max_pool2d(
            inputs,
            kernelSize,
            stride=strides,
            padding='VALID'
        )

    def avgPooling(self, inputs, kernelSize, strides):
        return tf.contrib.layers.avg_pool2d(
            inputs,
            kernelSize,
            stride=strides,
            padding='VALID'
        )

    def sepConvMobileNet(self, features, kernel_size, out_filters, stride, _name, dilationFactor=1, pad='SAME'):
        with tf.variable_scope(_name):
            output = tf.contrib.layers.separable_conv2d(
                features,
                None,
                kernel_size,
                depth_multiplier=1,
                stride=stride,
                weights_initializer=self.weightInitializer(),
                normalizer_fn=self.normalizer_fn,
                normalizer_params=self.norm_params,
                activation_fn=self.activation,
                weights_regularizer=None,
                padding=pad,
                rate=dilationFactor,
                scope='dw'
            )
            output = tf.contrib.layers.conv2d(
                output,
                out_filters, [1, 1],
                stride=1,
                weights_initializer=self.weightInitializer(),
                normalizer_fn=self.normalizer_fn,
                normalizer_params=self.norm_params,
                activation_fn=self.activation,
                weights_regularizer=None,
                rate=dilationFactor,
                scope='pw'
            )
            return output

    def model(self):
        conv1 = self.conv2d(self.input, 8, 3, 2, 'conv1')

        conv2 = self.sepConvMobileNet(conv1, 3, 16, 1, "conv2")
        conv2 = self.avgPooling(conv2, 2, 2)

        conv3 = self.sepConvMobileNet(conv2, 3, 32, 1, "conv3")
        conv3 = self.avgPooling(conv3, 2, 2)

        conv4 = self.sepConvMobileNet(conv3, 3, 32, 1, "conv4")
        conv4 = self.avgPooling(conv4, 2, 2)

        conv5 = self.sepConvMobileNet(conv4, 3, 32, 1, "conv5", 2)
        conv6 = self.sepConvMobileNet(conv5, 3, 32, 1, "conv6", 4)

        o1 = tf.image.resize_bilinear(conv6, [60, 80])

        conv7 = self.sepConvMobileNet(o1, 3, 64, 1, "conv7", 1)

        o2 = tf.image.resize_bilinear(conv7, [240, 320])

        conv8 = self.sepConvMobileNet(o2, 3, 64, 1, "conv8", 1)

        conv9 = self.sepConvMobileNet(conv8, 3, 3, 1, "conv9", 1)

        return conv9


net = network()
model = net.model()
model = tf.identity(model, name="model")

if __name__ == '__main__':
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs/graphs', sess.graph)
