import tensorflow as tf
from ImageProcessing import ImageLoader

import os

def saveGeneratedImages(sample_dir, batch_size, sample_images, generated_images, step):
        output_images = sample_images[0]
        for i in range(1, batch_size):
            output_images = tf.concat([output_images, sample_images[i]], axis=0)
        for label, images in generated_images.items():
            predicted_images = images[0]
            for i in range(1, batch_size):
                predicted_images = tf.concat([predicted_images, images[i]], axis=0)
            output_images = tf.concat([output_images, predicted_images], axis=1)

        output_images = (output_images * 0.5 + 0.5) * 255
        output_images = tf.cast(output_images, tf.uint8)
        sample_path = os.path.join(sample_dir, '{}-images.jpg'.format(step))
        image_bytes = tf.io.encode_jpeg(output_images)
        tf.io.write_file(sample_path, image_bytes)


generator_path = 'models/110000_G'
G = tf.keras.models.load_model(generator_path)
batch_size = 8

imageLoader = ImageLoader(batch_size=batch_size, image_size=(128, 128))
dataset = imageLoader.loadImages(dir_path='rp_imgs/', mode='test')

sample_images, sample_classes = next(iter(dataset.take(1)))
generated_images = {}
for label in range(4):
    trg_vec = tf.cast(tf.one_hot(label, depth=4), tf.float32)
    trg_vec = tf.reshape(trg_vec, (1, tf.shape(trg_vec)[0]))
    trg_vec = tf.tile(trg_vec, (tf.shape(sample_images)[0], 1))
    generated_images[label] = G(sample_images, trg_vec, training = False)
    
saveGeneratedImages(sample_dir='output', batch_size=batch_size, sample_images=sample_images, generated_images=generated_images, step=3)