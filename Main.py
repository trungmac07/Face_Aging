import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import argparse

from FaceAgingModel import FaceAgingStarGAN
from FaceAgingModel import StarGANConfig
from ImageProcessing import ImageLoader

def main(config):

    starGAN_config = StarGANConfig(config)
    starGAN = FaceAgingStarGAN(starGAN_config)

    if(config.mode != "train"):
        if config.result_dir and not os.path.exists(config.result_dir):
            os.makedirs(config.result_dir)
        gpu = tf.config.list_physical_devices('GPU')
        if gpu:
            print("Running on GPU")
            with tf.device('/GPU:0'):
                starGAN.generate_img(config.test_img_path, config.result_dir)
        else:
            print("Running on CPU")
            starGAN.generate_img(config.test_img_path, config.result_dir)

    else:
        imageLoader = ImageLoader(batch_size=16, image_size=(config.image_size, config.image_size))
        dataset = imageLoader.loadImages(dir_path='AAF_4/train/', mode='train')
        # Create directories if not exist.
        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)
        if not os.path.exists(config.model_save_dir):
            os.makedirs(config.model_save_dir)
        if not os.path.exists(config.sample_dir):
            os.makedirs(config.sample_dir)


        
        gpu = tf.config.list_physical_devices('GPU')
        if gpu:
            print("Running on GPU")
            with tf.device('/GPU:0'):
                starGAN.train(dataset)
        else:
            print("Running on CPU")
            starGAN.train(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    
    parser.add_argument('--chan_dim', type=int, default=4, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    
    # Training configuration.
    parser.add_argument('--num_steps', type=int, default=100000, help='number of total steps for training')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--start_step', type=int, default=0, help='resume training from this step')
    
    
    # Miscellaneous.
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    
    # Directories.
    parser.add_argument('--image_dir', type=str, default="./AAF_4/train/", help='image data directory')
    parser.add_argument('--log_dir', type=str, default='stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='stargan/models', help = 'model directory for loading and saving in training')
    parser.add_argument('--sample_dir', type=str, default='stargan/samples')
    parser.add_argument('--result_dir', type=str, default=None)
    parser.add_argument('--G_path', type=str, default='./model/G', help='Generator path for loading to test')
    parser.add_argument('--D_path', type=str, default='./model/D', help='Discriminator path for loading to test')
    parser.add_argument('--test_img_path', nargs='+', default=['test_img/man.jpg'], help='An image path for testing')


    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)

   

    config = parser.parse_args()
    
    main(config)