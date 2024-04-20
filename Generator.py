import tensorflow as tf
import numpy as np


class ResidualBlock(tf.keras.Model):
    def __init__(self, dim_out):
        super().__init__()

        self.r_layers = tf.keras.Sequential()
        self.r_layers.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
        self.r_layers.add(tf.keras.layers.Conv2D(filters=dim_out,
                                               kernel_size=3,
                                               strides=1,
                                               use_bias=False))
        self.r_layers.add(tf.keras.layers.BatchNormalization(axis=3))
        # self.r_layers.add(tf.keras.layers.BatchNormalization(axis=[0,3]))
        self.r_layers.add(tf.keras.layers.ReLU())
        self.r_layers.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
        self.r_layers.add(tf.keras.layers.Conv2D(filters=dim_out,
                                               kernel_size=3,
                                               strides=1,
                                               use_bias=False))
        self.r_layers.add(tf.keras.layers.BatchNormalization(axis=3))
        # self.r_layers.add(tf.keras.layers.BatchNormalization(axis=[0,3]))

    def call(self, x, training = True):
        return x + self.r_layers(x, training = training )

class Generator(tf.keras.Model):
    def __init__(self, image_size=128, conv_dim=64, chan_dim=5):
        super().__init__()

        self.g_layers = tf.keras.Sequential()
        self.g_layers.add(tf.keras.Input(shape=(image_size, image_size, 3+chan_dim)))
        self.g_layers.add(tf.keras.layers.ZeroPadding2D(padding=(3, 3)))
        self.g_layers.add(tf.keras.layers.Conv2D(filters=conv_dim,
                                               kernel_size=7,
                                               strides=1,
                                               use_bias=False))
        self.g_layers.add(tf.keras.layers.BatchNormalization(axis=3))
        # self.r_layers.add(tf.keras.layers.BatchNormalization(axis=[0,3]))
        self.g_layers.add(tf.keras.layers.ReLU())

        # Down-sampling layers
        curr_dim = conv_dim
        for i in range(2):
            self.g_layers.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
            self.g_layers.add(tf.keras.layers.Conv2D(filters=curr_dim*2,
                                                kernel_size=4,
                                                strides=2,
                                                use_bias=False))
            self.g_layers.add(tf.keras.layers.BatchNormalization(axis=3))
            # self.r_layers.add(tf.keras.layers.BatchNormalization(axis=[0,3]))
            self.g_layers.add(tf.keras.layers.ReLU())
            curr_dim *= 2

        # # Bottleneck layers
        for i in range(6):
            self.g_layers.add(ResidualBlock(dim_out=curr_dim))

        # # Up-sampling layers
        for i in range(2):
            # self.g_layers.add(tf.keras.layers.ZeroPadding2D(padding=1))
            self.g_layers.add(tf.keras.layers.Conv2DTranspose(filters=curr_dim//2,
                                                              kernel_size=4,
                                                              padding='same',
                                                              strides=2,
                                                              use_bias=False))
            self.g_layers.add(tf.keras.layers.BatchNormalization(axis=3))
            # self.r_layers.add(tf.keras.layers.BatchNormalization(axis=[0,3]))
            self.g_layers.add(tf.keras.layers.ReLU())
            curr_dim = curr_dim // 2

        # # Output layer
        self.g_layers.add(tf.keras.layers.ZeroPadding2D(padding=(3, 3)))
        self.g_layers.add(tf.keras.layers.Conv2D(filters=3,
                                                    kernel_size=7,
                                                    strides=1,
                                                    use_bias=False,
                                                    activation='tanh'))

    def call(self, x, c, training = True):
        c = tf.reshape(c, (tf.shape(c)[0], 1, 1, tf.shape(c)[1]))
        # print(c.shape)
        # print(c)
        c = tf.tile(c, (1, tf.shape(x)[1], tf.shape(x)[2], 1))

        # x = tf.reshape(x, (1, tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2])) # For testing
        # print(x.shape)
        x = tf.concat((x, c), axis=3)

        return self.g_layers(x, training = training)
    
    def generate_img(self, x, c):
        if(len(x.shape) < 4):
            x = tf.expand_dims(x, 0)
        img = self.call(x,c,training=False)[0]
        img = img * 0.5 + 0.5
        return img