import numpy as np
from neural_arithmatic_logic_unit import NALU
import tensorflow as tf


def generate_ds(count, low, high):
    x = np.random.randint(low, high, (count, 2))
    y = np.zeros((count, 4), dtype=np.float32)
    y[:, 0] = x[:, 0] + x[:, 1]
    y[:, 1] = x[:, 0] - x[:, 1]
    y[:, 2] = x[:, 0] * x[:, 1]
    y[:, 3] = x[:, 0] / x[:, 1]
    return x.astype(np.float32), y


if __name__ == "__main__":
    train_x, train_y = generate_ds(1000, 1, 10)
    test_x, test_y = generate_ds(100, 10, 100)
    model = tf.keras.models.Sequential()
    model.add(NALU(4))
    model.compile("rmsprop", tf.losses.mse)
    model.fit(train_x, train_y, epochs=2 ** 15, batch_size=128, validation_data=[test_x, test_y])
