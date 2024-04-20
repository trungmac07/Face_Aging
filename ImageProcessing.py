import tensorflow as tf
import numpy as np
import os

class ImageLoader():
    def __init__(self, batch_size, image_size):
        self.batch_size = batch_size
        self.image_size = image_size

    def load(self, image_file):
        # Read and decode an image file to a uint8 tensor
        image = tf.io.read_file(image_file)
        image = tf.io.decode_jpeg(image)

        # Convert image to float32 tensors
        image = tf.cast(image, tf.float32)

        return image

    def normalize(self, image):
        # Normalizing the images to [-1, 1]
        image = (image / 127.5) - 1
        return image

    def preprocessImage(self, image, training = True):
        # Resize
        image = tf.image.resize(image, self.image_size,
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # image = crop(image)

        if training and tf.random.uniform(()) > 0.5:
            # Random mirroring
            image = tf.image.flip_left_right(image)

        image = self.normalize(image)

        return image[:,:,0:3]


    def loadImages(self, dir_path, mode):
        image_set_list = []
        image_count = 0
        class_count = 0

        print('Loading images for {} mode...'.format(mode))
        l_d = os.listdir(dir_path).copy()
        for entry in sorted(l_d):
            print('Class {} - {}:\t'.format(class_count, entry), end='')
            entry = os.path.join(dir_path, entry)
            if os.path.isdir(entry):
                image_set = tf.data.Dataset.list_files(entry + '/*.jpg')
                image_set = image_set.map(lambda image_path: (self.preprocessImage(self.load(image_path)), class_count),
                                                  num_parallel_calls=tf.data.AUTOTUNE)

                num_images = image_set.cardinality().numpy()
                print(num_images)

                image_count += num_images

                image_set_list.append(image_set)

                class_count += 1

        print('Total images: {}'.format(image_count))

        def parseImageEntry(image, label):
            onehotvec = tf.one_hot(label, depth=class_count)
            return image, tf.cast(onehotvec, tf.float32)

        full_image_set = image_set_list[0]
        for image_set in image_set_list[1:]:
            full_image_set = full_image_set.concatenate(image_set)

        full_image_set = full_image_set.map(parseImageEntry,
                                            num_parallel_calls=tf.data.AUTOTUNE)
        full_image_set = full_image_set.shuffle(image_count)
        full_image_set = full_image_set.batch(self.batch_size)

        return full_image_set