import tensorflow as tf

import numpy as np

class SourceDiscriminator(tf.keras.Model):
    def __init__(self, image_size=128, conv_dim=64, chan_dim=5):
        super().__init__()

        self.d_layers = tf.keras.Sequential()
        self.d_layers.add(tf.keras.Input(shape=(image_size, image_size, 3)))
        self.d_layers.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
        self.d_layers.add(tf.keras.layers.Conv2D(filters=conv_dim,
                                               kernel_size=4,
                                               strides=2))
        self.d_layers.add(tf.keras.layers.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, 6):
            self.d_layers.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
            self.d_layers.add(tf.keras.layers.Conv2D(filters=curr_dim*2,
                                                kernel_size=4,
                                                strides=2))
            self.d_layers.add(tf.keras.layers.LeakyReLU(0.01))
            curr_dim *= 2

        self.src_layer = tf.keras.Sequential()
        self.src_layer.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
        self.src_layer.add(tf.keras.layers.Conv2D(filters=1,
                                            kernel_size=3,
                                            strides=1,
                                            use_bias=False))

    def call(self, x, training = True):
        z = self.d_layers(x)

        src_res = self.src_layer(z, training = training)

        return src_res

class ClassDiscriminator(tf.keras.Model):
    def __init__(self, image_size=128, conv_dim=64, chan_dim=5):
        super().__init__()

        self.d_layers = tf.keras.Sequential()
        self.d_layers.add(tf.keras.Input(shape=(image_size, image_size, 3)))
        self.d_layers.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
        self.d_layers.add(tf.keras.layers.Conv2D(filters=conv_dim,
                                               kernel_size=4,
                                               strides=2))
        self.d_layers.add(tf.keras.layers.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, 6):
            self.d_layers.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
            self.d_layers.add(tf.keras.layers.Conv2D(filters=curr_dim*2,
                                                kernel_size=4,
                                                strides=2))
            self.d_layers.add(tf.keras.layers.LeakyReLU(0.01))
            curr_dim *= 2

        kernel_size = int(image_size / np.power(2, 6))
        self.cls_layer = tf.keras.Sequential()
        self.cls_layer.add(tf.keras.layers.Conv2D(filters=chan_dim,
                                            kernel_size=kernel_size,
                                            strides=1,
                                            use_bias=False))
        # self.cls_layer.add(tf.keras.layers.Softmax(axis=1))
        self.cls_layer.add(tf.keras.layers.Softmax())

    def call(self, x, training = True):
        z = self.d_layers(x)

        cls_res = self.cls_layer(z, training = training)

        return tf.reshape(cls_res, (tf.shape(cls_res)[0], tf.shape(cls_res)[3]))