# IOU
def box_iou_xyxy(box1, box2):
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)
    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    inter_h = np.maximum(ymax - ymin + 1., 0.)
    inter_w = np.maximum(xmax - xmin + 1., 0.)
    intersection = inter_h * inter_w
    union = s1 + s2 - intersection
    iou = intersection / union
    return iou
