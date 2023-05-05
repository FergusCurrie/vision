import tensorflow as tf
import numpy as np
from pascal import get_pascal_tfds
import warnings
from training import keras_train

warnings.filterwarnings("ignore")


def down_step(features, filters):
    x = tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="valid")(features)
    x = tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="valid")(x)
    p = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    return x, p


def up_step(features, filters, skip_connections, crop_size):
    cropped_skip_connections = tf.keras.layers.Cropping2D(crop_size)(skip_connections)
    x = tf.keras.layers.Conv2DTranspose(filters, kernel_size=(2, 2), strides=2)(features)
    x = tf.keras.layers.concatenate([x, cropped_skip_connections])
    x = tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="valid")(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="valid")(x)
    return x


def bottle_neck(features, filters):
    x = tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="valid")(features)
    x = tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="valid")(x)
    return x


def unet():
    input_layer = tf.keras.layers.Input(shape=(572, 572, 3))
    skip1, x = down_step(input_layer, 64)
    skip2, x = down_step(x, 128)
    skip3, x = down_step(x, 256)
    skip4, x = down_step(x, 512)
    x = bottle_neck(x, 1024)
    x = up_step(x, 512, skip4, crop_size=4)
    x = up_step(x, 256, skip3, crop_size=16)
    x = up_step(x, 128, skip2, crop_size=40)
    x = up_step(x, 64, skip1, crop_size=88)
    preds = tf.keras.layers.Conv2D(21, (1, 1))(x)
    unet = tf.keras.Model(inputs=input_layer, outputs=preds)
    return unet


def crop_labels(image, label):
    cropped_labels = tf.keras.layers.Cropping2D(92)(label)
    return image, cropped_labels


if __name__ == "__main__":
    image_size = 572
    batch_size = 16
    train_tfds, val_tfds = get_pascal_tfds(image_size=image_size, batch_size=batch_size)
    train_tfds = train_tfds.map(crop_labels)
    val_tfds = val_tfds.map(crop_labels)
    unet = unet()
    keras_train(unet, train_tfds, val_tfds, epochs=10, batch_size=batch_size, name="unet")
