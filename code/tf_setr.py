"""
TODO:
- learned positional encoding.
- saving checkpoints 
- validation loss 
- tensorflow config
"""
import tensorflow as tf
from pascal import get_pascal_tfds
import warnings
from transformer import MyTransformer
from training import keras_train


warnings.filterwarnings("ignore")


def image_sequentialisation(image, label):
    """
    # TODO : batch dim?
    Convert image into sequence: f : R^(H x W x 3) -> R^(n x -1) # n flattened patches
    L = HW / 256

    So essentially divid image into 16x16 grid. -> input should be dividsible by 16

    """
    # start with loopy implementation
    steps = image_size // 16
    # patches = tf.zeros((batch_size, steps * steps, 16 * 16 * 3))
    patches = []
    for i in range(steps):
        for j in range(steps):
            patch = image[:, i * 16 : (i + 1) * 16, j * 16 : (j + 1) * 16, :]
            flatten_patch = tf.reshape(patch, shape=(batch_size, -1))
            patches.append(flatten_patch)
    patches = tf.stack(patches)
    sequence = tf.reshape(patches, shape=(batch_size, steps * steps, 16 * 16 * 3))
    return sequence, label


def linear_projection(inputs):
    x = tf.keras.layers.Dense(d_model)(inputs)
    return x


def stem(inputs):
    x = linear_projection(inputs)
    # TODO: positional encoding
    return x, None


def decoder(inputs):
    """
    Essentially need to reshape from (HW/256 x C) to (H x W x C)
    Naive upsampling:
    1. (1x1conv) -> (bn) -> (relu) -> (1x1conv) to get to N classes
    2. bilinear upsampling to get to H x W
    """
    x = tf.reshape(inputs, shape=(batch_size, image_size // 16, image_size // 16, d_model))
    x = tf.keras.layers.Conv2D(256, (1, 1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(21, (1, 1))(x)
    x = tf.keras.layers.UpSampling2D(size=(16, 16), interpolation="bilinear")(x)
    return x


def build_setr(inputs):
    # output of encoder is (batch_size, 256, 512) or (batch_size, )
    transformer_inputs = tf.keras.Input(shape=(seq_len, d_model))

    encoder1 = MyTransformer(seq_len=seq_len, d_model=d_model, batch_size=batch_size).build_encoder(transformer_inputs)
    encoder2 = MyTransformer(seq_len=seq_len, d_model=d_model, batch_size=batch_size).build_encoder(transformer_inputs)

    # tf.keras.Model(inputs=inputs, outputs=transformer_encoder_output)
    proj, positional_encoding = stem(inputs)
    x = encoder1(proj)
    x = encoder2(x)
    x = decoder(x)
    x = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


image_size = 512
seq_len = 1024  # 32 * 32 for (image_size=512)//16 patches
d_model = 512
batch_size = 16

inputs = tf.keras.Input(
    shape=(
        seq_len,
        16 * 16 * 3,
    )
)
setr = build_setr(inputs)

train_tfds, val_tfds = get_pascal_tfds(image_size=image_size, batch_size=batch_size)
train_tfds = train_tfds.map(image_sequentialisation)
val_tfds = val_tfds.map(image_sequentialisation)


keras_train(setr, train_tfds, val_tfds, batch_size, epochs=10, name="setr")
