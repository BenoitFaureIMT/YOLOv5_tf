import tensorflow as tf

class YOLOv5(object):
    def __init__(self, tf_lite_f_path):
        self.interpreter = tf.lite.Interpreter(tf_lite_f_path)
        print("Input : ", self.interpreter.get_input_details())
        print("Output : ", self.interpreter.get_output_details())

        

YOLOv5("test.tflite")