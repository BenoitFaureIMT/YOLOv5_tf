import cv2

img = cv2.imread("test.jpg")
coor = [[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8]]

class DrawBB(image,coords):
    def __init__(self):
        self.image = image
        self.coords = coords
        for i in coords:
            x,y,w,h = cv2.boundingRect(i)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,0), 4)
        cv2.imshow('BBox', image)

