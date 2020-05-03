import tensorflow as tf
class Preprocessor():
    def _load_data(self):
        (train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
        return (train_images, train_labels)

    def _preprocess_dataset(self, tf_dataset, batch_size, buffer_size=60000):
        batched_dataset = tf_dataset.shuffle(buffer_size).map(
            lambda image, label: self._preprocess_images(image, label)).batch(batch_size, drop_remainder=True).prefetch(1)
        return batched_dataset

    @tf.function
    def _preprocess_images(self, image, label):
        image = tf.cast(image, tf.float32)
        expanded_image = tf.expand_dims(image, axis=-1)
        normalised_image = (expanded_image - 127.5) / 127.5
        return normalised_image, label

    def get_preprocessed_train_data(self, batch_size=256):
        (train_images, train_labels) = self._load_data()
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_dataset = self._preprocess_dataset(train_dataset, batch_size)
        return train_dataset