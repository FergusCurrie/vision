import tensorflow as tf
import numpy as np

"""
TODO:
- override keras class 
- testing 
- masking 
- what is d_k used for 
"""

"""

    pos = np.arange(seq_len)
    d = np.arange(d_model)

    # Compute the scaling factor
    scaling_factor = 1 / (10000 ** (2 * d / d_model))

    # Compute the outer product of pos and scaling_factor
    outer_product = np.outer(pos, scaling_factor)

    # Compute the sinusoidal and cosine values using broadcasting
    sinusoidal = np.sin(outer_product[:, :, np.arange(d_model) % 2 == 0])
    cosine = np.cos(outer_product[:, :, np.arange(d_model) % 2 == 1])

    # Combine the sinusoidal and cosine values into P
    P = np.zeros((seq_len, d_model))
    P[:, np.arange(d_model) % 2 == 0] = sinusoidal
    P[:, np.arange(d_model) % 2 == 1] = cosine

    # Repeat P for each sequence in the batch
    P = np.tile(P[None, :, :], (batch_size, 1, 1))

    """


class MyTransformer:
    def __init__(self, seq_len, d_model, batch_size):
        # super(MyTransformer, self).__init__()
        self.d_k = 1
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.d_model = d_model
        self.h = 8
        # self.positional_encoding = self.get_positional_encoding()

    def feed_forward_network(self, inputs):
        x = tf.keras.layers.Dense(2048, activation="relu")(inputs)
        x = tf.keras.layers.Dense(512)(x)
        return x  # (batch_size, seq_len, d_model)

    def get_positional_encoding(self):
        """
        Result

        Args:
            seq_len (_type_): _description_

        Returns:
            P : positional encoder. P in R^(seq_len, d_model)
        """
        # TODO: non-loopy impl
        P = np.zeros((self.batch_size, self.seq_len, self.d_model))
        for i in range(self.seq_len):
            for j in range(self.seq_len):
                # P same accross batch
                Z = [i / (10000 ** (2 * j / self.d_model)) for _ in range(self.batch_size)]
                if j % 2 == 0:
                    P[:, i, j] = np.sin(Z)
                else:
                    P[:, i, j] = np.cos(Z)

        return tf.convert_to_tensor(P)

    def add_norm(self, path1, path2):
        return tf.keras.layers.LayerNormalization()(path1 + path2)

    def linear_projection(self, inputs):
        x = tf.keras.layers.Dense(self.d_model)(inputs)
        return x

    def multi_head_attention(self, q, k, v, mask=None):
        """
        q, k, v: (..., seq_len, depth)
        mask: (..., seq_len, seq_len)
        """
        x = []
        for _ in range(self.h):
            proj_q = self.linear_projection(q)
            proj_k = self.linear_projection(k)
            proj_v = self.linear_projection(v)
            x.append(self.scaled_dot_product_attention(proj_q, proj_k, proj_v))
        x = tf.concat(x, axis=-1)
        return self.linear_projection(x)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul = tf.matmul(q, tf.transpose(k, perm=[0, 2, 1]))  # transpoe h x w
        scale = matmul / np.sqrt(self.d_model)
        # masking shoudl
        # if not mask is None:
        #    masked = mask * scale  # TODO: check this is correct way to mask
        softmax = tf.keras.activations.softmax(scale)
        return tf.matmul(softmax, v)

    def encoder(self, inputs, mask=None):
        input_embedding = self.linear_projection(inputs)
        inputs = input_embedding

        mha = self.multi_head_attention(v=inputs, k=inputs, q=inputs)  # ?
        sub_layer = self.add_norm(mha, inputs)

        ffn = self.feed_forward_network(sub_layer)
        encoder_outputs = self.add_norm(ffn, sub_layer)
        return encoder_outputs

    def decoder(self, inputs, encoder_outputs, mask=None):
        # masked multi-head self attention
        mha = self.multi_head_attention(v=inputs, k=inputs, q=inputs)  # ?
        sub_layer = self.add_norm(mha, inputs)

        # masked multi-head attention w encoder
        mha = self.multi_head_attention(v=encoder_outputs, k=encoder_outputs, q=sub_layer)  # ?
        sub_layer = self.add_norm(mha, sub_layer)

        ffn = self.feed_forward_network(sub_layer)
        decoder_output = self.add_norm(ffn, sub_layer)
        return decoder_output

    def stem(self, inputs, mask=None):
        input_embedding = self.linear_projection(inputs)
        # TODO: positional encoding
        inputs = input_embedding
        return inputs

    def transformer(self, inputs):
        """

        Args:
            inputs (tensor): R^(n x 3072) # 32 x32 x 3  patches
        """
        # Prepare inputs
        input_embedding = self.linear_projection(inputs)
        # inputs = tf.add([input_embedding, self.positional_encoding])
        inputs = input_embedding

        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(inputs, encoder_outputs)
        proj = self.linear_projection(decoder_outputs)
        probabilities = tf.keras.activations.softmax(proj)
        return probabilities

    def build(self, inputs):
        transformer_output = self.transformer(inputs)
        return tf.keras.Model(inputs=inputs, outputs=transformer_output)

    def build_encoder(self, inputs):
        transformer_encoder_output = self.encoder(inputs)
        return tf.keras.Model(inputs=inputs, outputs=transformer_encoder_output)


if __name__ == "__main__":
    seq_len = 16 * 16
    d_model = 512
    batch_size = 8

    inputs = tf.keras.Input(
        shape=(
            seq_len,
            d_model,
        )
    )
    t_encoder = MyTransformer(seq_len=seq_len, d_model=d_model, batch_size=batch_size).build_encoder(inputs)
    print(t_encoder.summary())
