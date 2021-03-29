from keras.layers import *

def GCN(x, out_channels, kernel_size, IMAGE_ORDERING='channels_first'):
    conv11 = Conv2D(out_channels, (kernel_size, 1), data_format=IMAGE_ORDERING, padding='same')(x)
    conv12 = Conv2D(out_channels, (1, kernel_size), data_format=IMAGE_ORDERING, padding='same')(conv11)

    conv21 = Conv2D(out_channels, (1, kernel_size), data_format=IMAGE_ORDERING, padding='same')(x)
    conv22 = Conv2D(out_channels, (kernel_size, 1), data_format=IMAGE_ORDERING, padding='same')(conv21)

    return conv12 + conv22