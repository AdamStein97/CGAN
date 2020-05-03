from cgan.utils import generate_image, load_config
import os
import tensorflow as tf
import cgan
from cgan.model.discriminator import Discriminator
from cgan.model.generator_model import init_generator_model

config = load_config('config.yaml')

discriminator = Discriminator(**config['discriminator_config'])
generator = init_generator_model(**config['generator_config'])

model_dir = os.path.join(cgan.MODEL_DIR, "trained_model")
checkpoint = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator)

checkpoint.restore(model_dir).expect_partial()

image = generate_image(generator, digit=9, show=True)