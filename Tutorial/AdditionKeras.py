import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import dataGen as data
import matplotlib.pyplot as plt


"""
inputs = keras.Input(shape=(3,))

dense = keras.layers.Dense()
nextLayer = dense(inputs)
nextLayer = keras.layers.Dense(64)(nextLayer)
outputs = keras.layers.Dense(1)(nextLayer)
"""

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(3,)),
    keras.layers.Dense(36, activation="relu"),
    keras.layers.Dense(36, activation="relu"),
    keras.layers.Dense(1)
])

model.summary()
learning_params = keras.optimizers.Adam(learning_rate=0.0003)
model.compile(optimizer=learning_params,loss="mse", metrics=["mae"])

history = model.fit(data.training_eq, data.training_targets, epochs=6, batch_size=40, validation_split=0.2, verbose=2)

test_loss, test_acc = model.evaluate(data.test_eq, data.test_targets)

print('Test accuracy:', test_acc)

model.save('additionModel_5.h5')

a= np.array([[2000,0,3000],[4,1,5]])

print(model.predict(a))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()