import tensorflow as tf
class InputNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, coding_sample_dim=100, coding_embedded_channels=128, labelling_embedded_channels=1,
                 labelling_embedding_size=50, noise_encoding_activation='relu', conditional_encoding_activation='relu', **kwargs):
        super(InputNoiseLayer, self).__init__()
        self.coding_sample_dim = coding_sample_dim
        self.coding_embedded_channels = coding_embedded_channels
        self.labelling_embedded_channels = labelling_embedded_channels
        self.coding_layer_size = coding_embedded_channels * 7 * 7
        self.conditional_layer_size = labelling_embedded_channels * 7 * 7
        self.labelling_embedding_size = labelling_embedding_size
        self.noise_encoding_activation = noise_encoding_activation
        self.conditional_encoding_activation = conditional_encoding_activation


    def build(self, input_shape):
        self.noise_layer = tf.keras.layers.Dense(self.coding_layer_size, activation=self.noise_encoding_activation, trainable=True, input_shape=input_shape)
        self.conditional_layer = tf.keras.layers.Dense(self.conditional_layer_size, activation=self.conditional_encoding_activation, trainable=True)
        self.noise_reshape = tf.keras.layers.Reshape((7, 7, self.coding_embedded_channels))
        self.label_reshape = tf.keras.layers.Reshape((7, 7,  self.labelling_embedded_channels))
        self.label_embed_layer = tf.keras.layers.Embedding(10, self.labelling_embedding_size)

    def call(self, label):
        noise_sample = tf.random.normal(shape=[tf.shape(label)[0], self.coding_sample_dim])
        noise_encoding = self.noise_layer(noise_sample)
        noise_encoding = self.noise_reshape(noise_encoding)
        label_embed = self.label_embed_layer(label)
        label_encoding = self.conditional_layer(label_embed)
        label_encoding = self.label_reshape(label_encoding)
        z = tf.concat([noise_encoding, label_encoding], axis=-1)
        return z

def init_generator_model(convolution_layers=2, convolution_kernels=128, convolution_kernel_size=(4, 4), convolution_stride=(2, 2),
                         convolution_padding='same', convolution_activation='LeakyReLU', **kwargs):
    conv_layers = []
    for i in range(convolution_layers):
        conv_layers.append(
            tf.keras.layers.Conv2DTranspose(convolution_kernels, convolution_kernel_size, strides=convolution_stride,
                                   padding=convolution_padding))
        conv_layers.append(tf.keras.layers.BatchNormalization())
        conv_layers.append(getattr(tf.keras.layers, convolution_activation)())
    return tf.keras.models.Sequential([InputNoiseLayer()] +
                                       conv_layers +
                                       [tf.keras.layers.Conv2DTranspose(1, (7, 7), padding='same', activation='tanh')])
