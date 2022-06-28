def IoU(box1, box2) :
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    width = (x2 - x1)
    height = (y2 - y1)
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    area_combined = area1 + area2 - area_overlap
    iou = area_overlap / area_combined
    return iou

b1 = [0.1,0.1,0.4,0.4]
b2 = [0.1,0.1,0.4,0.4]

print(IoU(b1, b2))