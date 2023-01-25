import tensorflow as tf
import time
import random
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0

new_model = tf.keras.models.load_model('mymodel.h5')
new_model.summary()
new_model.evaluate(x_test,  y_test, verbose=2)

