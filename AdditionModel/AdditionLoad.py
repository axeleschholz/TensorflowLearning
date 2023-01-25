import tensorflow as tf
mnist = tf.keras.datasets.mnist
import dataGen as data


new_model = tf.keras.models.load_model('additionModel.h5')
new_model.summary()
new_model.evaluate(data.test_eq,  data.test_targets, verbose=2)

a = new_model.predict([[1,1,2], [10,0,4], [100,0,199]])
print(a)