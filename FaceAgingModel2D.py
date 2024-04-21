import tensorflow as tf

import os
import time
import datetime
import matplotlib.pyplot as plt

from Discriminator2D import SourceDiscriminator, ClassDiscriminator
from Generator import Generator

class FaceAgingStarGAN():
    def __init__(self, config):
        self.config = config
        self.constructModel()

    def constructModel(self):
        if (self.config.start_step is None or self.config.start_step == 0):
            self.SD = SourceDiscriminator(image_size=self.config.image_size, conv_dim=self.config.sd_conv_dim, chan_dim=self.config.chan_dim)
            self.CD = ClassDiscriminator(image_size=self.config.image_size, conv_dim=self.config.cd_conv_dim, chan_dim=self.config.chan_dim)
            self.G = Generator(image_size=self.config.image_size, conv_dim=self.config.g_conv_dim, chan_dim=self.config.chan_dim)

            self.sd_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.sd_lr, beta_1=self.config.beta_1, beta_2=self.config.beta_2)
            self.cd_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.cd_lr, beta_1=self.config.beta_1, beta_2=self.config.beta_2)
            self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.g_lr, beta_1=self.config.beta_1, beta_2=self.config.beta_2)

            print('Constructed new models')
        else:
            SD_path = os.path.join(self.config.model_save_dir, '{}_SD'.format(self.config.start_step))
            CD_path = os.path.join(self.config.model_save_dir, '{}_CD'.format(self.config.start_step))
            G_path = os.path.join(self.config.model_save_dir, '{}_G'.format(self.config.start_step))

            self.SD = tf.keras.models.load_model(SD_path)
            self.CD = tf.keras.models.load_model(CD_path)
            self.G = tf.keras.models.load_model(G_path)

            self.sd_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.sd_lr, beta_1=self.config.beta_1, beta_2=self.config.beta_2)
            self.cd_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.cd_lr, beta_1=self.config.beta_1, beta_2=self.config.beta_2)
            self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.g_lr, beta_1=self.config.beta_1, beta_2=self.config.beta_2)

            print('Loaded the trained models from step {}'.format(self.config.start_step))

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
            src_res = self.SD(real_imgs)
            cls_res = self.CD(real_imgs)
            d_loss_real = - tf.reduce_mean(src_res)
            d_loss_cls = tf.keras.losses.CategoricalCrossentropy()(org_vec, cls_res)

            # Compute loss with fake images
            with tape.stop_recording(): # detach???
                fake_imgs = self.G(real_imgs, trg_vec)
            src_res = self.SD(fake_imgs) # detach???
            cls_res = self.CD(fake_imgs)
            d_loss_fake = tf.reduce_mean(src_res)

            # Compute loss for gradient penalty
            alpha = tf.random.uniform(shape=[tf.shape(real_imgs)[0], 1, 1, 1], minval=0., maxval=1.)
            x_hat = alpha * real_imgs + (1 - alpha) * fake_imgs
            x_hat = tf.Variable(x_hat, trainable=True)
            src_res = self.SD(x_hat)

            dydx = tape.gradient(target=src_res,
                                 sources=x_hat)
            dydx = tf.reshape(dydx, (dydx.shape[0], -1))
            dydx_l2norm = tf.sqrt(tf.reduce_sum(dydx**2, axis=1))
            d_loss_gp = tf.reduce_mean(tf.square(dydx_l2norm - 1.0))

            # Backward and optimize
            sd_loss = d_loss_real + d_loss_fake + self.config.lambda_gp * d_loss_gp
            cd_loss = self.config.lambda_cls * d_loss_cls
            # d_loss = d_loss_real + d_loss_fake + self.config.lambda_cls * d_loss_cls + self.config.lambda_gp * d_loss_gp
            d_loss = sd_loss + cd_loss

            # Logging.
            if log == True:
                loss_log = {}
                loss_log['D/loss_real'] = d_loss_real
                loss_log['D/loss_fake'] = d_loss_fake
                loss_log['D/loss_cls'] = d_loss_cls
                loss_log['D/loss_gp'] = d_loss_gp
                loss_log['D/loss_sd'] = sd_loss
                loss_log['D/loss_cd'] = cd_loss
                loss_log['D/loss_total'] = d_loss

            if (step + 1) % self.config.n_critic == 0:
                # Original-to-target domain.
                fake_imgs = self.G(real_imgs, trg_vec)
                src_res = self.SD(fake_imgs)
                cls_res = self.CD(fake_imgs)
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
        sd_gradients = tape.gradient(sd_loss, self.SD.trainable_variables)
        self.sd_optimizer.apply_gradients(zip(sd_gradients, self.SD.trainable_variables))

        cd_gradients = tape.gradient(cd_loss, self.CD.trainable_variables)
        self.cd_optimizer.apply_gradients(zip(cd_gradients, self.CD.trainable_variables))

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
                SD_path = os.path.join(self.config.model_save_dir, '{}_SD'.format(cur_step + 1))
                CD_path = os.path.join(self.config.model_save_dir, '{}_CD'.format(cur_step + 1))
                G_path = os.path.join(self.config.model_save_dir, '{}_G'.format(cur_step + 1))
                self.SD.save(SD_path)
                self.CD.save(CD_path)
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