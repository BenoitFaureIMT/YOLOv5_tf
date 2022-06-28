import tensorflow as tf

class YOLOv5(object):
    def __init__(self, folder_path):
        self.model = tf.keras.models.load_model(folder_path)
        self.model.summary()

YOLOv5("saved_model")