import tensorflow as tf

def weight_variable(shape, stddev=0.1, name="weight"):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W, b):
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        conv_2d_b = tf.nn.bias_add(conv_2d, b)
        return conv_2d_b

def GCN(x, filter_size, in_channels, out_channels, stddev=0.1):
    """
    Global Convolution Network (Spatial attention)
    :param x: Input features, (batch_size, H, W, C_in)
    :param filter_size: kernel size
    :param in_channels: input_channels
    :param out_channels: output_channels
    :param stddev: stddev
    :return: spatial map (batch_size, H, W, C_out), C_out=1 for Spatial attention map without sigmoid
    """
    with tf.name_scope('GCN'):
        w11 = weight_variable([filter_size, 1, in_channels, out_channels], stddev, name="w11")
        w12 = weight_variable([1, filter_size, out_channels, out_channels], stddev, name="w12")
        b11 = bias_variable([out_channels], name="b11")
        b12 = bias_variable([out_channels], name="b12")
        conv11 = conv2d(x, w11, b11)
        conv12 = conv2d(conv11, w12, b12)

        w21 = weight_variable([1, filter_size, in_channels, out_channels], stddev, name="w21")
        w22 = weight_variable([filter_size, 1, out_channels, out_channels], stddev, name="w22")
        b21 = bias_variable([out_channels], name="b21")
        b22 = bias_variable([out_channels], name="b22")
        conv21 = conv2d(x, w21, b21)
        conv22 = conv2d(conv21, w22, b22)

        gcn_out = conv12 + conv22

        return gcn_out


def ChannelAttention(input_feature, ratio=2):
    """
    channel-wise attention
    :param input_feature: (batch_size, H, W, C)
    :param ratio: channel -> channel//ratio -> channel
    :return: (batch_size, 1, 1, C) without sigmoid
    """
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.name_scope('channel_att'):
        channel = input_feature.get_shape()[-1]
        avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)

        assert avg_pool.get_shape()[1:] == (1, 1, channel)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_0',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_1',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel)

        max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   name='mlp_0',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel,
                                   name='mlp_1',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)

        #scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')

        return avg_pool + max_pool #input_feature * scale

if __name__ == '__main__':
    in_x = tf.random.normal([4, 10, 10, 32], mean=-1, stddev=4)
    print(in_x.shape)
    gcn_x = GCN(in_x, 7, 32, 1)
    ca_x = ChannelAttention(in_x)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        out_gcn = sess.run(gcn_x)
        print(out_gcn.shape)
        out_ca = sess.run(ca_x)
        print(out_ca.shape)
