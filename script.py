from PIL import Image
import numpy as np
import tensorflow as tf


image = Image.open('test.jpg')
data = np.asarray(image)
print(data.shape)


interpreter = tf.lite.Interpreter("test.tflite")
interpreter.allocate_tensors()

#input_index = interpreter.get_input_details()#[0]["index"]
#output_index = interpreter.get_output_details()#[0]["index"]


interpreter.set_tensor(interpreter.get_input_details()[0]['index'], data)
interpreter.invoke()

output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
print(output_data)