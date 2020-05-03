import random
import matplotlib.pyplot as plt
import numpy as np
import cgan
import os
import yaml

def load_yaml(file_dir):
    with open(file_dir, 'rb') as file:
        dict = yaml.load(file, Loader=yaml.FullLoader)
    return dict


def load_config(config_filename='config.yaml'):
    config = load_yaml(os.path.join(cgan.CONFIG_DIR, config_filename))
    discriminator_config = load_yaml(os.path.join(cgan.CONFIG_DIR, config['discriminator_config']))
    generator_config = load_yaml(os.path.join(cgan.CONFIG_DIR, config['generator_config']))
    config['discriminator_config'] = discriminator_config
    config['generator_config'] = generator_config
    return config


def generate_image(generator, digit=None, show=True, save_file_name="generated_img.png"):
    if digit is None:
        digit = random.randint(0,9)

    generated = generator(np.expand_dims(digit, axis=0), training=False)

    generated_image = generated[0, :, :, 0] * 127.5 + 127.5

    if show:
        plt.imshow(generated_image, cmap='gray')
        plt.savefig(os.path.join(cgan.IMG_DIR, save_file_name))
        plt.show()

    return generated_image