import tensorflow as tf
import time as time
import os
from cgan.model.loss_functions import generator_loss, discriminator_loss
from cgan.model.discriminator import Discriminator
from cgan.model.generator_model import init_generator_model
import cgan

class Trainer():
    @tf.function
    def train_step(self, batch, generator, discriminator, generator_optimizer, discriminator_optimizer):
        images, labels = batch
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(labels, training=True)

            real_output = discriminator(images, labels, training=True)
            fake_output = discriminator(generated_images, labels, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    def train(self, dataset, epochs=200, model_name="cgan", generator_lr=1e-4, discriminator_lr=1e-4, checkpoint_save_freq=15):
        generator_optimizer = tf.keras.optimizers.Adam(generator_lr)
        discriminator_optimizer = tf.keras.optimizers.Adam(discriminator_lr)

        discriminator = Discriminator()
        generator = init_generator_model()

        checkpoint_prefix = os.path.join(cgan.MODEL_DIR, "{}_final_ckpt".format(model_name))
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator=generator,
                                         discriminator=discriminator)

        for epoch in range(epochs):
            start = time.time()

            for batch in dataset:
                self.train_step(batch, generator, discriminator, generator_optimizer, discriminator_optimizer)

            if (epoch + 1) % checkpoint_save_freq == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        return discriminator, generator