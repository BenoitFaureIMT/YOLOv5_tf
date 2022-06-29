def iou(b1,b2):
    x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    y_overlap = max(0, min(b1[1], b2[1]) - max(b1[3], b2[3]))
    intersection = x_overlap * y_overlap

    area1 = (b1[2] - b1[0]) * (b1[1] - b1[3])
    area2 = (b2[2] - b2[0]) * (b2[1] - b2[3])
    union = area1 + area2 - intersection

    return intersection / union

print(iou([0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4]))