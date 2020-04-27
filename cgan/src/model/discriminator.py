import tensorflow as tf
class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.convolution_layers = tf.keras.models.Sequential([tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
                                            tf.keras.layers.BatchNormalization(),
                                            tf.keras.layers.LeakyReLU(),
                                            tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
                                            tf.keras.layers.BatchNormalization(),
                                            tf.keras.layers.LeakyReLU(),
                                            tf.keras.layers.Flatten()])
    self.label_embed_layer = tf.keras.layers.Embedding(10, 50)
    self.label_encoding_layer = tf.keras.layers.Dense(784, activation='relu')
    self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    self.label_reshape_layer =  tf.keras.layers.Reshape((28, 28 ,1))

  def call(self, image, label):
    label_embed = self.label_embed_layer(label)
    label_encode = self.label_encoding_layer(label_embed)
    label_reshape = self.label_reshape_layer(label_encode)
    combined = tf.concat([image, label_reshape], axis=-1)
    image_encode = self.convolution_layers(combined)
    return self.output_layer(image_encode)

