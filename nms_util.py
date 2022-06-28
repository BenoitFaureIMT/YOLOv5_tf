import numpy as np
import tensorflow as tf
"""
def IoU(b1, b2):
    Ix = max(b1[0], b2[0])
    Iy = max(b1[1], b2[1])
    Ixp = min(b1[0] + b1[2], b2[0] + b2[2])
    Iyp = min(b1[1] + b1[3], b2[1] + b2[3])
    I = (Ixp - Ix) * (Iyp - Iy)
    return 1 / ( (b1[2] * b1[3] + b2[2] * b2[3]) / I - 1)
"""
def IoU(b1,b2):
    
    inter = np.logical_and([b1[0],b1[1],b1[0]+b1[2],b1[1]+b1[3]],[b2[0],b2[1],b2[0]+b2[2],b2[1]+b2[3]])
    union = np.logical_or([b1[0],b1[1],b1[0]+b1[2],b1[1]+b1[3]],[b2[0],b2[1],b2[0]+b2[2],b2[1]+b2[3]])
    print(np.sum(inter)/np.sum(union))
    return np.sum(inter)/np.sum(union)

def NMS(boxes, classes, scores):
    clBoxes = []
    clClasses = []
    clScores = []
    while(len(scores) > 0):

        clScores.append(max(scores))
        i = scores.index(clScores[-1])
        clBoxes.append(boxes[i])
        clClasses.append(classes[i])
        del boxes[i]
        del classes[i]
        del scores[i]

        j = 0
        for _ in range(len(scores)):
            iou = IoU(boxes[i], clBoxes[-1])
            if iou > 0.5:
                del boxes[j]
                del classes[j]
                del scores[j]
            else:
                j += 1
    return clBoxes, clClasses, clScores