import tensorflow as tf
import tensorflow.keras.backend as K


class NALU(tf.keras.layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(NALU, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.w_hat = self.add_variable(
            name="w_hat",
            shape=(input_dim, self.output_dim),
            trainable=True
        )

        self.m_hat = self.add_weight(
            name="m_hat",
            shape=(input_dim, self.output_dim),
            trainable=True
        )

        self.big_g = self.add_weight(
            name="big_g",
            shape=(input_dim, self.output_dim),
            trainable=True
        )

    def call(self, inputs, **kwargs):
        W = K.tanh(self.w_hat) * K.sigmoid(self.m_hat)
        a = K.dot(inputs, W)
        m = K.exp(K.dot(K.log(K.abs(inputs) + K.epsilon()), W))
        g = K.sigmoid(K.dot(inputs, self.big_g))
        y = g * a + (1 - g) * m
        return y

