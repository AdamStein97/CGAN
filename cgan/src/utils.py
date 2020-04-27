import random
import matplotlib.pyplot as plt
import numpy as np
import cgan
import os

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