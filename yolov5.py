import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

#Ours
from nms_util import NMS

class YOLOv5(object):
    def __init__(self, tf_lite_f_path):
        self.interpreter = tf.lite.Interpreter(tf_lite_f_path)
    
    def run_img(self, img_path):
        #Get interpreter info
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.allocate_tensors()

        #Get image
        img = Image.open(img_path).resize((640, 640), Image.ANTIALIAS)
        img = tf.keras.preprocessing.image.img_to_array(img)

        #Set image and run
        self.interpreter.set_tensor(input_details[0]['index'], tf.expand_dims(img, axis=0))
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        return output_data
    
    #----------------------------------------------------------------Code from Glenn Jocher----------------------------------------------------------------
    def classFilter(self, classdata):
        classes = []  # create a list
        for i in range(classdata.shape[0]):         # loop through all predictions
            classes.append(classdata[i].argmax())   # get the best classification location
        return classes  # return classes (int)

    def analyse_output(self, output_data):  # input = interpreter, output is boxes(xyxy), classes, scores
        output_data = output_data[0]                # x(1, 25200, 7) to x(25200, 7)
        boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
        scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
        classes = self.classFilter(output_data[..., 5:]) # get classes
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
        xywh = [[x[i], y[i], w[i], h[i]] for i in range(len(x))]
        #xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

        return xywh, classes, scores  # output is boxes(x,y,x,y), classes(int), scores(float) [predictions length]
    #-----------------------------------------------------------------------------------------------------------------------------------------------------

    def detect(self, img_path, min_score = 0.1):
        xywh, classes, scores = self.analyse_output(self.run_img(img_path))
        scores = list(scores)

        xywh, classes, scores = NMS(xywh, classes, scores)

        i = 0
        for _ in range(len(scores)):
            if(scores[i] < min_score):
                del xywh[i]
                del classes[i]
                del scores[i]
            else:
                i += 1
                
        return xywh, classes, scores
    
    def display(xywh, classes, scores, img_path):
        img = cv2.imread(img_path)
        coords = xywh

        for i in coords:
            x,y,w,h = i[0]*640,i[1]*640,i[2]*640,i[3]*640
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255,0,0), 1)

        cv2.imshow('BBox', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

yolo_model = YOLOv5("test.tflite")
xywh, classes, scores = yolo_model.detect("test.jpg")
yolo_model.display(xywh, classes, scores, "test.jpg")


