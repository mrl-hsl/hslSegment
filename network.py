import tensorflow as tf


class network():
    def __init__(self):
        self.input = tf.placeholder(
            tf.float32, [None, 240, 320, 3], name='input')
        self.isTraning = tf.placeholder(tf.bool, [], name='isTraning')

        self.activation = tf.nn.leaky_relu
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

    def spp(self, input):
        with tf.variable_scope("spp"):
            # prymid1 = tf.layers.conv2d(input, 24, 1, dilation_rate=(1,1), activation=tf.nn.relu6, padding="same",name = "p1")
            # prymid1 = self.sepConvMobileNet(input, 24, 1, dilation_rate=(1,1), activation=tf.nn.relu6, padding="same",name = "p1")
            # prymid2 = tf.layers.separable_conv2d(input, 24, 3, dilation_rate=(6,6), activation=tf.nn.relu6, padding="same",name = "p2")
            # prymid3 = tf.layers.separable_conv2d(input, 24, 3, dilation_rate=(12,12), activation=tf.nn.relu6, padding="same",name = "p3")
            # prymid4 = tf.layers.separable_conv2d(input, 24, 3, dilation_rate=(18,18), activation=tf.nn.relu6, padding="same",name = "p4")
            prymid1 = self.sepConvMobileNet(input, 3, 24, 1, "p1", pad='SAME')
            prymid2 = self.sepConvMobileNet(input, 3, 24, 1, "p2", pad='SAME')
            prymid3 = self.sepConvMobileNet(input, 3, 24, 1, "p3", pad='SAME')
            prymid4 = self.conv2d(input, 24, 1, 1, 'p4')

            sppConcat = tf.concat([prymid1,prymid2,prymid3,prymid4],3,name="sppConcat")

            return sppConcat

    def model(self):
        conv1 = self.conv2d(self.input, 6, 3, 2, 'conv1')

        conv2 = self.sepConvMobileNet(conv1, 3, 12, 1, "conv2")
        conv2 = self.avgPooling(conv2, 2, 2)

        conv3 = self.sepConvMobileNet(conv2, 3, 12, 1, "conv3")
        conv3 = self.avgPooling(conv3, 2, 2)

        spp = self.spp(conv3)

        pooling = self.conv2d(conv1, 12, 1, 1, 'pooling')

        spp_merg = self.conv2d(spp, 48, 1, 1, 'spp-merg')
        o1 = tf.image.resize_bilinear(spp_merg, [120, 160])

        concat = tf.concat([o1, pooling], 3, name="concat")

        o2 = self.sepConvMobileNet(concat, 3, 3, 1, "o2", 1)

        out = tf.image.resize_bilinear(o2, [240, 320])

        return out


net = network()
model = net.model()
model = tf.identity(model, name="model")

if __name__ == '__main__':
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs/graphs', sess.graph)
