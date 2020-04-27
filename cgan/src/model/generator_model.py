import tensorflow as tf
class InputNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, coding_sample_dim=100, coding_layer_size=6272, conditional_layer_size=49):
        super(InputNoiseLayer, self).__init__()
        self.coding_sample_dim = coding_sample_dim
        self.coding_layer_size = coding_layer_size
        self.conditional_layer_size = conditional_layer_size

    def build(self, input_shape):
        self.noise_layer = tf.keras.layers.Dense(self.coding_layer_size, activation='relu', trainable=True)
        self.conditional_layer = tf.keras.layers.Dense(self.conditional_layer_size, activation='relu', trainable=True)
        self.noise_reshape = tf.keras.layers.Reshape((7, 7, 128))
        self.label_reshape = tf.keras.layers.Reshape((7, 7, 1))
        self.label_embed_layer = tf.keras.layers.Embedding(10, 50)

    def call(self, label):
        noise_sample = tf.random.normal(shape=[tf.shape(label)[0], self.coding_sample_dim])
        noise_encoding = self.noise_layer(noise_sample)
        noise_encoding = self.noise_reshape(noise_encoding)
        label_embed = self.label_embed_layer(label)
        label_encoding = self.conditional_layer(label_embed)
        label_encoding = self.label_reshape(label_encoding)
        z = tf.concat([noise_encoding, label_encoding], axis=-1)
        return z

def init_generator_model():
    return tf.keras.models.Sequential([InputNoiseLayer(),
                                        tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.LeakyReLU(),
                                        tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.LeakyReLU(),
                                        tf.keras.layers.Conv2DTranspose(1, (7, 7), padding='same', activation='tanh')
                                        ])