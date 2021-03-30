from keras.layers import *

def GCN(x, out_channels, kernel_size):
    """
    Global Convolution Network (Spatial attention)
    :param x: Input features, (batch_size, H, W, C_in)
    :param out_channels: output_channels
    :param kernel_size: kernel size
    :return:spatial map (batch_size, H, W, out_channels), out_channels=1 for Spatial attention map without sigmoid
    """
    conv11 = Conv2D(out_channels, (kernel_size, 1), padding='same')(x)
    conv12 = Conv2D(out_channels, (1, kernel_size), padding='same')(conv11)

    conv21 = Conv2D(out_channels, (1, kernel_size), padding='same')(x)
    conv22 = Conv2D(out_channels, (kernel_size, 1), padding='same')(conv21)

    return conv12 + conv22


def channel_attention(input_feature, channels, ratio=2):
    """
    channel wise attention
    :param input_feature: (batch_size, h, w, c)
    :param channels: channels are same as c of input_feature
    :param ratio: channel -> channel//ratio -> channel
    :return: channel-wise attention vector without sigmoid,(batch_size, 1, 1, c)
    """
    shared_layer_one = Dense(channels // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channels,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channels))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    #assert avg_pool.shape[1:] == (1, 1, channels // ratio)
    avg_pool = shared_layer_two(avg_pool)
    #assert avg_pool.shape[1:] == (1, 1, channels)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channels))(max_pool)
    #assert max_pool._keras_shape[1:] == (1, 1, channels)
    max_pool = shared_layer_one(max_pool)
    #assert max_pool._keras_shape[1:] == (1, 1, channels // ratio)
    max_pool = shared_layer_two(max_pool)
    #assert max_pool._keras_shape[1:] == (1, 1, channels)

    cbam_feature = Add()([avg_pool, max_pool])
    #cbam_feature = Activation('sigmoid')(cbam_feature)

    # if K.image_data_format() == "channels_first":
    #     cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return cbam_feature#multiply([input_feature, cbam_feature])

if __name__ == '__main__':
    in_x = Input(shape=(10, 10, 32))
    out_gcn = GCN(in_x, out_channels=1, kernel_size=7)
    print(out_gcn.shape)
    out_ca = channel_attention(in_x, 32, 2)
    print(out_ca.shape)
