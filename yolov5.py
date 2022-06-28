import tensorflow as tf
from PIL import Image
import numpy as np

class YOLOv5(object):
    def __init__(self, tf_lite_f_path):
        self.interpreter = tf.lite.Interpreter(tf_lite_f_path)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.allocate_tensors()
        print("Input : ", input_details)
        print("Output : ", output_details)

        img = Image.open("chien.webp").resize((640, 640), Image.ANTIALIAS)
        img = tf.keras.preprocessing.image.img_to_array(img)

        self.interpreter.set_tensor(input_details[0]['index'], tf.expand_dims(img, axis=0))
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        print(output_data)


YOLOv5("test.tflite")