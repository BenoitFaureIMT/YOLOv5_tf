import os

import tensorflow as tf
from tensorflow import keras

new_model = tf.keras.models.load_model('/Users/nathanodic/Downloads/saved_model')

# Check its architecture
new_model.summary()