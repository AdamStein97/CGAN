import tensorflow as tf

class Discriminator(tf.keras.Model):
  def __init__(self, convolution_layers=2, convolution_kernels=128, convolution_kernel_size=(3,3),
               convolution_activation='LeakyReLU', convolution_stride=(2,2), convolution_padding='same',
               labelling_embedding_size=50, labelling_encoding_layer_size=784, labelling_encoding_layer_activation='relu'):
    super(Discriminator, self).__init__()

    conv_layers = []
    for i in range(convolution_layers):
        conv_layers.append(tf.keras.layers.Conv2D(convolution_kernels, convolution_kernel_size, strides=convolution_stride,
                               padding=convolution_padding))
        conv_layers.append(tf.keras.layers.BatchNormalization())
        conv_layers.append(getattr(tf.keras.layers, convolution_activation)())
    conv_layers.append(tf.keras.layers.Flatten())

    self.convolution_encoder = tf.keras.models.Sequential(conv_layers)
    self.label_embed_layer = tf.keras.layers.Embedding(10, labelling_embedding_size)
    self.label_encoding_layer = tf.keras.layers.Dense(labelling_encoding_layer_size, activation=labelling_encoding_layer_activation)
    self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    self.label_reshape_layer =  tf.keras.layers.Reshape((28, 28 ,1))

  def call(self, image, label):
    label_embed = self.label_embed_layer(label)
    label_encode = self.label_encoding_layer(label_embed)
    label_reshape = self.label_reshape_layer(label_encode)
    combined = tf.concat([image, label_reshape], axis=-1)
    image_encode = self.convolution_encoder(combined)
    return self.output_layer(image_encode)

