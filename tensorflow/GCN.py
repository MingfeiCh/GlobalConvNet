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

if __name__ == '__main__':
    pass