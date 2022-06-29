import tensorflow as tf
import cv2
import time

img = tf.keras.preprocessing.image.img_to_array(cv2.imread("test.jpg"))
model = tf.saved_model.load("/Users/nathanodic/Downloads/saved_model")

input_tensor = tf.convert_to_tensor(img)
input_tensor = input_tensor[tf.newaxis, ...]

start_time = time.time()
detection = model(input_tensor)
end_time = time.time()

print(end_time - start_time)
print(detection.shape)