# dqn policy
import numpy as np
import tensorflow as tf
from gym.utils import colorize


def dense_nn(inputs, layers_sizes, name="mlp", output_fn=None, dropout_keep_prob=None, batch_norm=False, training=True):

    print(colorize("Building mlp {} | sizes: {}".format(
        name, [inputs.shape[0]] + layers_sizes), "green"))

    model = tf.keras.Sequential()

    for i, layer in enumerate(layers_sizes):
        print("Layer:", name + '_l' + str(i), layer)
        if i > 0 and dropout_keep_prob is not None and training:
            model.add(tf.keras.layers.Dropout(dropout_keep_prob))

        model.add(tf.keras.layers.Dense(layer, activation='relu' if i < len(layers_sizes) - 1 else None,
                                        name=name + '_l' + str(i)))

        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())

    model.build(input_shape=(4,))
    return model


if __name__ == '__main__':
    myinputs = np.array([0.2, 0.4, 1.2, 0.4])
    layers = [16, 16]
    model1 = dense_nn(inputs=myinputs, layers_sizes=layers, batch_norm=True)
    print(model1.summary())
