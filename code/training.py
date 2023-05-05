"""
Good for debugging or for some more custom models. 
"""

import tensorflow as tf
import os
import pandas as pd


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


def train_loop(model, loss_object, metric, optimizer, tfds, unet):
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


def keras_train(model, train_tfds, val_tfds, batch_size, epochs, name):
    print(model.summary())
    adamw = tf.keras.optimizers.experimental.AdamW()
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    categorical_accuracy = tf.keras.metrics.CategoricalAccuracy()

    checkpoint_path = f"checkpoints/cp_{name}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_best_only=True)

    model.compile(loss=loss, optimizer=adamw, metrics=[categorical_accuracy])
    history = model.fit(
        x=train_tfds,
        steps_per_epoch=40,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_steps=100,
        validation_data=val_tfds,
        callbacks=[cp_callback],
    )

    history = pd.DataFrame(history.history)
    history.to_csv(f"history/{name}_history.csv", index=False)
