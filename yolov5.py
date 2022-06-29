from concurrent.futures import process
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import time

#Ours
from nms_util import NMS

class YOLOv5(object):
    def __init__(self, path):
        self.model = tf.saved_model.load(path)
    
    def load_tflite(self, tf_lite_f_path):
        self.interpreter = tf.lite.Interpreter(tf_lite_f_path)
    
    def run_tflite(self, img):
        #Get interpreter info
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.allocate_tensors()

        #Process image
        img = tf.keras.preprocessing.image.img_to_array(img)

        #Set image and run
        self.interpreter.set_tensor(input_details[0]['index'], tf.expand_dims(img, axis=0))
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        return output_data

    def run_net(self, img):
        #Process image
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.convert_to_tensor(img) / 225
        img = img[tf.newaxis, ...]

        output_data = self.model(img).numpy()

        return output_data
    
    def process_output(self, output_data, min_score = 0.5):

        #Output shape : (Excluding first dimension) -> [..., [x, y, w, h, confidence, ind0 conf, ind1 conf, ...], ...]
        #output_data = [[bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[4], bbox[5:].argmax()] for bbox in output_data[0]]
        #output_data = [[bbox[1] - bbox[3] / 2, bbox[0] - bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[0] + bbox[2] / 2, bbox[4], bbox[5:].argmax()] for bbox in output_data[0]]
        output_data = output_data[0]

        #Elimination
        # i = 0
        # for _ in range(len(output_data)):
        #     if(output_data[i][4] < min_score):
        #         np.delete(output_data, i) #del output_data[i]
        #     else:
        #         i += 1
        output_data = output_data[np.where(output_data[:, 4] > min_score)]
        
        # o = np.copy(output_data)
        # output_data[:, 1] = o[:, 0] - o[:, 2] / 2  # top left x
        # output_data[:, 0] = o[:, 1] - o[:, 3] / 2  # top left y
        # output_data[:, 3] = o[:, 0] + o[:, 2] / 2  # bottom right x
        # output_data[:, 2] = o[:, 1] + o[:, 3] / 2  # bottom right y
        output_data = [[bbox[1] - bbox[3] / 2, bbox[0] - bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[0] + bbox[2] / 2, bbox[4], bbox[5:].argmax()] for bbox in output_data]

        #NMS    
        #output_data = NMS(output_data)
        output_data = np.array(output_data)
        selected = tf.image.non_max_suppression(output_data[:, :4], output_data[:, 4], 30000, 0.65)
        output_data = output_data[selected, :]

        return output_data

    #Utility functions
    def run_img(self, img_path):
        return self.run_net(cv2.resize(cv2.imread(img_path), (640, 640), interpolation=cv2.INTER_LINEAR))
    
    def warm_up(self):
        self.run_net(tf.zeros((640, 640, 3)))
    
    #Interface functions
    def detect(self, img_path, min_score = 0.5):
        output_data = self.process_output(self.run_img(img_path), min_score)
                
        return output_data

    def display(self, output_data, img_path):
        img = cv2.imread(img_path)
        coords = output_data[..., :4]

        for i in coords:
            y,x,yp,xp = i[0]*640,i[1]*640,i[2]*640,i[3]*640
            cv2.rectangle(img, (int(x), int(y)), (int(xp), int(yp)), (255,0,0), 1)

        cv2.imshow('BBox', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

yolo_model = YOLOv5("saved_model")

yolo_model.warm_up()

img = cv2.resize(cv2.imread("test.jpg"), (640, 640), interpolation=cv2.INTER_LINEAR)
t = time.perf_counter()
output_data = yolo_model.process_output(yolo_model.run_net(img))
t -= time.perf_counter()
print(-t)

yolo_model.display(output_data, "test.jpg")