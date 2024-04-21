import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os 
import time
import datetime
from pathlib import Path

from ImageProcessing import ImageLoader
from Discriminator import Discriminator
from Generator import Generator



class FaceAgingStarGAN():
    def __init__(self, config):
        self.config = config
        self.constructModel()

    def constructModel(self):
        if(self.config.mode != "train"):
            self.G = Generator(image_size=self.config.image_size, conv_dim=self.config.g_conv_dim, chan_dim=self.config.chan_dim)
            self.D = Discriminator(image_size=self.config.image_size, conv_dim=self.config.d_conv_dim, chan_dim=self.config.chan_dim)
            
            self.G = tf.keras.models.load_model(self.config.G_path)
            self.D = tf.keras.models.load_model(self.config.D_path)
            print('Model loaded successfully for testing')
            
            
        else:
            if (self.config.start_step is None or self.config.start_step == 0):
                self.D = Discriminator(image_size=self.config.image_size, conv_dim=self.config.d_conv_dim, chan_dim=self.config.chan_dim)
                self.G = Generator(image_size=self.config.image_size, conv_dim=self.config.g_conv_dim, chan_dim=self.config.chan_dim)

                self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.d_lr, beta_1=self.config.beta_1, beta_2=self.config.beta_2)
                self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.g_lr, beta_1=self.config.beta_1, beta_2=self.config.beta_2)

                print('Constructed new models')
            else:
                try:
                    D_path = os.path.join(self.config.model_save_dir, '{}_D'.format(self.config.start_step))
                    G_path = os.path.join(self.config.model_save_dir, '{}_G'.format(self.config.start_step))

                    self.D = tf.keras.models.load_model(D_path)
                    self.G = tf.keras.models.load_model(G_path)

                    self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.d_lr, beta_1=self.config.beta_1, beta_2=self.config.beta_2)
                    self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.g_lr, beta_1=self.config.beta_1, beta_2=self.config.beta_2)
                    print('Loaded the trained models from step {} successfully'.format(self.config.start_step))

                except:
                    print('Could not load the trained models. Models now is not trained')

    def createTensorBoard(self):
        self.summary_writer = tf.summary.create_file_writer(self.config.log_dir)

    def plotGeneratedImages(self, sample_images, sample_classes, generated_images):
        fig, ax = plt.subplots(nrows=3, ncols=self.config.chan_dim+1, figsize=(15,15))
        for i in range(3):
            ax[i][0].imshow(sample_images[i] * 0.5 + 0.5)
            ax[i][0].title.set_text(str(sample_classes[i]))
        for label, images in generated_images.items():
            for i in range(3):
                ax[i][label+1].imshow(images[i] * 0.5 + 0.5)
                ax[i][label+1].title.set_text(str(label))

    def saveGeneratedImages(self, sample_images, generated_images, step):
        output_images = sample_images[0]
        for i in range(1, self.config.batch_size):
            output_images = tf.concat([output_images, sample_images[i]], axis=0)
        for label, images in generated_images.items():
            predicted_images = images[0]
            for i in range(1, self.config.batch_size):
                predicted_images = tf.concat([predicted_images, images[i]], axis=0)
            output_images = tf.concat([output_images, predicted_images], axis=1)

        output_images = (output_images * 0.5 + 0.5) * 255
        output_images = tf.cast(output_images, tf.uint8)
        sample_path = os.path.join(self.config.sample_dir, '{}-images.jpg'.format(step))
        image_bytes = tf.io.encode_jpeg(output_images)
        tf.io.write_file(sample_path, image_bytes)

    # @tf.function
    def train_step(self, real_imgs, org_vec, step, log=False):
        # Preprocess data
        rand_idx = tf.random.shuffle(tf.range(tf.shape(org_vec)[0]))
        trg_vec = tf.gather(org_vec, rand_idx)

        with tf.GradientTape(persistent=True) as tape:
            # Train the discriminator
            # Compute loss with real images
            src_res, cls_res = self.D(real_imgs)
            d_loss_real = - tf.reduce_mean(src_res)
            d_loss_cls = tf.keras.losses.CategoricalCrossentropy()(org_vec, cls_res)

            # Compute loss with fake images
            with tape.stop_recording(): # detach???
                fake_imgs = self.G(real_imgs, trg_vec)
            src_res, cls_res = self.D(fake_imgs) # detach???
            d_loss_fake = tf.reduce_mean(src_res)

            # Compute loss for gradient penalty
            alpha = tf.random.uniform(shape=[tf.shape(real_imgs)[0], 1, 1, 1], minval=0., maxval=1.)
            x_hat = alpha * real_imgs + (1 - alpha) * fake_imgs
            x_hat = tf.Variable(x_hat, trainable=True)
            src_res, _ = self.D(x_hat)

            dydx = tape.gradient(target=src_res,
                                 sources=x_hat)
            dydx = tf.reshape(dydx, (dydx.shape[0], -1))
            dydx_l2norm = tf.sqrt(tf.reduce_sum(dydx**2, axis=1))
            d_loss_gp = tf.reduce_mean(tf.square(dydx_l2norm - 1.0))

            # Backward and optimize
            d_loss = d_loss_real + d_loss_fake + self.config.lambda_cls * d_loss_cls + self.config.lambda_gp * d_loss_gp

            # Logging.
            if log == True:
                loss_log = {}
                loss_log['D/loss_real'] = d_loss_real
                loss_log['D/loss_fake'] = d_loss_fake
                loss_log['D/loss_cls'] = d_loss_cls
                loss_log['D/loss_gp'] = d_loss_gp
                loss_log['D/loss_total'] = d_loss

            if (step + 1) % self.config.n_critic == 0:
                # Original-to-target domain.
                fake_imgs = self.G(real_imgs, trg_vec)
                src_res, cls_res = self.D(fake_imgs)
                g_loss_fake = - tf.reduce_mean(src_res)
                g_loss_cls = tf.keras.losses.CategoricalCrossentropy()(trg_vec, cls_res)

                # Target-to-original domain.
                rec_imgs = self.G(fake_imgs, org_vec)
                g_loss_rec = tf.reduce_mean(tf.abs(real_imgs - rec_imgs))

                # Backward and optimize
                g_loss = g_loss_fake + self.config.lambda_rec * g_loss_rec + self.config.lambda_cls * g_loss_cls

                # Logging.
                if log == True:
                    loss_log['G/loss_fake'] = g_loss_fake
                    loss_log['G/loss_rec'] = g_loss_rec
                    loss_log['G/loss_cls'] = g_loss_cls
                    loss_log['G/loss_total'] = g_loss

        # Calculate and apply the gradients for generator and discriminator
        d_gradients = tape.gradient(d_loss, self.D.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.D.trainable_variables))

        if (step + 1) % self.config.n_critic == 0:
            g_gradients = tape.gradient(g_loss, self.G.trainable_variables)
            self.g_optimizer.apply_gradients(zip(g_gradients, self.G.trainable_variables))

        # Miscellaneous
        return loss_log if log == True else None

    def train(self, dataset):
        start_time = time.time()

        sample_images, sample_classes = next(iter(dataset.take(1)))

        self.createTensorBoard()

        total_steps = self.config.num_steps + (self.config.start_step if self.config.start_step is not None else 0)

        print("Start training from step {}".format(self.config.start_step if self.config.start_step is not None else 0))

        for step, (real_imgs, org_vec) in dataset.repeat().take(self.config.num_steps).enumerate():

            cur_step = step + (self.config.start_step if self.config.start_step is not None else 0)

            # Training and loggging
            if (cur_step + 1) % self.config.log_step == 0:
                loss_log = self.train_step(real_imgs=real_imgs, org_vec=org_vec, step=cur_step, log=True)

                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, cur_step + 1, total_steps)
                for tag, value in loss_log.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                with self.summary_writer.as_default():
                    for tag, value in loss_log.items():
                        tf.summary.scalar(name=tag, data=value, step=cur_step + 1)

            else:
                self.train_step(real_imgs=real_imgs, org_vec=org_vec, step=cur_step, log=False)

            # Save model checkpoints.
            if (cur_step + 1) % self.config.model_save_step == 0:
                D_path = os.path.join(self.config.model_save_dir, '{}_D'.format(cur_step + 1))
                G_path = os.path.join(self.config.model_save_dir, '{}_G'.format(cur_step + 1))
                self.D.save(D_path)
                self.G.save(G_path)
                print('Saved models into {}'.format(self.config.model_save_dir))

            # Generate sample images at different ages
            if (cur_step + 1) % self.config.sample_step == 0:
                generated_images = {}
                for label in range(self.config.chan_dim):
                    trg_vec = tf.cast(tf.one_hot(label, depth=self.config.chan_dim), tf.float32)
                    trg_vec = tf.reshape(trg_vec, (1, tf.shape(trg_vec)[0]))
                    trg_vec = tf.tile(trg_vec, (tf.shape(sample_images)[0], 1))
                    generated_images[label] = self.G(sample_images, trg_vec, training = False)

                self.saveGeneratedImages(sample_images=sample_images, generated_images=generated_images, step=cur_step+1)

    def generate_img(self, test_img_path = [], result_dir = None):
        
        img_loader = ImageLoader(16, (128,128))
        for path in test_img_path:
            base_path = ""
            if(result_dir):
                base_path = os.path.join(result_dir, Path(path).stem)
            else:
                base_path = os.path.join(os.path.dirname(path), Path(path).stem)
            print(base_path)
            test_img = img_loader.load(path)
            test_img = img_loader.preprocessImage(test_img, training = False)  

            for i in range(self.config.chan_dim):
                trg_vec = tf.one_hot(i, depth=self.config.chan_dim)
                trg_vec = tf.reshape(trg_vec, (1, tf.shape(trg_vec)[0]))
                if(len(test_img.shape) < 4):
                    test_img = tf.expand_dims(test_img, 0)
                #print(test_img.shape)
                new_img = self.G(test_img,trg_vec,training = False)[0]
                new_img = new_img * 0.5 + 0.5
                pil_img = tf.keras.preprocessing.image.array_to_img(new_img)
                pil_img.save(base_path + "_" + str(i) + ".jpg" )
                print(base_path + "_" + str(i) + ".jpg" )
               


            

class StarGANConfig(object):
    def __init__(self, args):                        
        
        #  Model configuration.
        self.chan_dim = args.chan_dim
        self.image_size = args.image_size
        self.g_conv_dim = 64
        self.d_conv_dim = 64
        self.lambda_cls = 1.5 
        self.lambda_rec = 10.9
        self.lambda_gp = 10.1

        # Training configuration.
        self.batch_size = 16
        self.num_steps = args.num_steps
        self.g_lr = 0.00009
        self.d_lr = 0.00009
        self.n_critic = args.n_critic
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.start_step = args.start_step

        # Step size
        self.log_step = args.log_step
        self.sample_step = args.sample_step
        self.model_save_step = args.model_save_step

        # Directories.
        self.image_dir = args.image_dir
        self.log_dir = args.log_dir
        self.model_save_dir = args.model_save_dir
        self.sample_dir = args.sample_dir
        self.D_path = args.D_path
        self.G_path = args.G_path

        # Miscellaneous.
        self.mode = args.mode