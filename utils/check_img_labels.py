import glob
import numpy as np
import cv2
from yolo_utils import xywh2xyxy

def cv_show(img):
    cv2.imshow('',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for img_name in glob.glob('./Data/VOC2007/train/images/*'):
    label_path = img_name.replace('images','labels').replace('jpg','txt')
    with open(label_path,'r') as f:
        l = [x.split() for x in f.read().strip().splitlines()]
        l = np.array(l,dtype=np.float32)
    
    img = cv2.imread(img_name)
    size = img.shape[:2]
    for box in l:
        (xy1,xy2) = xywh2xyxy(box[1:],size[1],size[0])
        xy1 = np.array(xy1,dtype=int)
        xy2 = np.array(xy2,dtype=int)
        cv2.rectangle(img,xy1,xy2,color = (255,0,0),thickness=3)

    cv_show(img)
