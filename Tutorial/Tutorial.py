#https://www.tensorflow.org/tutorials/quickstart/beginner
import tensorflow as tf
import numpy
import pandas as pd

mnist = tf.keras.datasets.mnist

#Download and scale values to range between 0 and 1
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#Build model with one input/output tensor on each layer
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

#create training data predictions 
predictions = model(x_train[:1]).numpy()

#convert matrix into probabilities
tf.nn.softmax(predictions).numpy()


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.evaluate(x_test,  y_test, verbose=2)



model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

model.summary()

model.save('mymodel.h5')
