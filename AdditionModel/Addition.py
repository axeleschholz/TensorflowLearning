import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras

x = tf.placeholder(dtype = tf.float32, shape = [None, 3])
y = tf.placeholder(dtype = tf.int32, shape=[None])

inputs = keras.Input(shape=(4,))

dense = keras.layers.Dense(64)
nextLayer = dense(inputs)
nextLayer = keras.layers.Dense(64)(nextLayer)
outputs = keras.layers.Dense(1)(nextLayer)

model = keras.Model(inputs=inputs, outputs=outputs, name="addition_model")
model.summary()

