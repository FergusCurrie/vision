"""
TODO:
- learned positional encoding.
- saving checkpoints 
- validation loss 
- tensorflow config
"""
import os
import tensorflow as tf
import numpy as np
from pascal import PascalDataGenerator
import warnings
from transformer import MyTransformer
import pandas as pd


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


def train_step(images, labels, model, loss_object, metric, optimizer):
    print(tf.math.reduce_mean(images))
    print(tf.math.reduce_mean(labels))
    with tf.GradientTape() as tape:
        logits = model(images)
        print(tf.math.reduce_mean(logits))
        loss = loss_object(labels, logits)
        print(tf.math.reduce_mean(loss))
        # print(f"loss = {tf.math.reduce_mean(loss)}")
        metric.update_state(labels, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    # print(tf.math.reduce_mean(grads))
    print("GRADS MEAN CALC")
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train_loop(model, loss_object, metric, optimizer, tfds):
    training_steps = 120
    tfds = tfds.batch(1)
    for epoch in range(50):
        images, labels = next(iter(tfds))
        images = images[0]
        labels = labels[0]
        loss = train_step(images, labels, model, loss_object, metric, optimizer)

        # if i > training_steps:
        #     break
        # i += 1

        print(f"epoch{epoch}: loss = {loss:.2f} accuracy metric = {metric.result():.2f}")


# tf.debugging.enable_check_numerics(stack_height_limit=30, path_length_limit=50)


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

# TRAIN
pdg = PascalDataGenerator(image_size=image_size, batch_size=batch_size)
train_tfds = tf.data.Dataset.from_generator(
    pdg.data_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([batch_size, image_size, image_size, 3], [batch_size, image_size, image_size, 21]),
)
train_tfds = train_tfds.map(image_sequentialisation)  # .batch(batch_size)


# VAL
pdg = PascalDataGenerator(image_size=image_size, batch_size=batch_size, train_test_val="val")
val_tfds = tf.data.Dataset.from_generator(
    pdg.data_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([batch_size, image_size, image_size, 3], [batch_size, image_size, image_size, 21]),
)
val_tfds = val_tfds.map(image_sequentialisation)

# adam = tf.keras.optimizers.Adam(clipvalue=0.1)
adamw = tf.keras.optimizers.experimental.AdamW()  # TODO: remove
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
categorical_acc = tf.keras.metrics.CategoricalAccuracy()
iou = tf.keras.metrics.MeanIoU(num_classes=21)

setr = build_setr(inputs)


# train_loop(setr, loss, metric, adamw, tfds)

checkpoint_path = "checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_best_only=True)


setr.compile(loss=loss, optimizer=adamw, metrics=[categorical_acc])
history = setr.fit(
    x=train_tfds,
    steps_per_epoch=40,
    batch_size=batch_size,
    epochs=10,
    verbose=1,
    validation_steps=100,
    validation_data=val_tfds,
    callbacks=[cp_callback],
)

history_df = pd.DataFrame(history.history)

# Save the history to a CSV file
history_df.to_csv("history/setr_history.csv", index=False)

# manual trianing


# testing pushing data through for nan
# batch = next(iter(tfds))
# batch[0].shape, batch[1].shape
# print(np.unique(batch[1]))
# pred = setr(batch[0])
# print(tf.math.reduce_mean(pred))
# print(tf.math.reduce_mean(batch[1]))
# print(pred.shape)
# print(batch[1].shape)
# l = loss(y_true=batch[1], y_pred=pred)
# print(l)
