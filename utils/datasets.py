import torch
from torch.utils.data import Dataset
import os
import cv2 as cv
import numpy as np
import random
import math
from torch.utils.data import DataLoader

def cv_show(img):
    cv.imshow('',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

class My_dataset(Dataset):
    def __init__(self,path,size,augmentation = False):
        self.img_size = size
        self.img_path = path
        self.img_path_list = [self.img_path + os.sep + x for x in os.listdir(self.img_path)]
        self.label_path_list = ['txt'.join(x.replace(os.sep+'images'+os.sep, os.sep+'labels'+os.sep, 1).rsplit(x.split('.')[-1], 1)) for x in self.img_path_list]
        self.augmentation = augmentation

    def __getitem__(self, index):

        with open(self.label_path_list[index],'r') as f:
            l = [x.split() for x in f.read().strip().splitlines()]
            l = np.array(l,dtype = np.float32)

        nl = len(l)  #label的长度
        label = torch.zeros((nl,6))

        if nl:
            label[:, 1:] = torch.from_numpy(l)

        img = cv.imread(self.img_path_list[index])
        
        if self.augmentation:
            img,label = self.random_flip(img,label)
            img = self.random_blur(img)
            img = self.random_Erasing(img)
            img = self.random_brightness(img)
            img = self.random_hue(img)
            img = self.random_saturation(img)
            img,label = self.rand_crop(img,label)

        img = cv.resize(img, (self.img_size,self.img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
            
        return torch.from_numpy(img),label
    
    def __len__(self):
        return len(self.img_path_list)
            

    def random_flip(self,img,box):
        '''
        随机翻转图片
        '''
        if random.random() < 0.5:
            return img,box

        img = np.fliplr(img)
        if box.shape[0] != 0:
            box[:,2] = 1 - box[:,2]

        return img,box

    def random_blur(self,img):
        '''
        随机给图片进行滤波
        '''
        if random.random() < 0.5:
            return img

        ksize = random.choice([2,3,4,5])
        img = cv.blur(img,(ksize,ksize))

        return img

    def random_Erasing(self,img,p1=0.5,s_l=0.02,s_h=0.3,r1=0.3,r2=0.5):
        if random.random() < p1:
            return img

        h,w = img.shape[:2]
        s = h * w    #原始图像面积

        while True:
            s_e = random.uniform(s_l,s_h) * s  #擦除框的面积
            r_e = random.uniform(r1,r2)  #擦除框的长宽比

            H_e = int(math.sqrt(s_e * r_e))   #擦除框的高度
            W_e = int(math.sqrt(s_e / r_e))   #擦除框的宽度

            x_e = random.randint(0,w)   #擦除框的左上角x坐标
            y_e = random.randint(0,h)   #左上角y坐标
            
            if x_e + W_e <= w and y_e + H_e <= h:
                img[y_e:y_e+H_e,x_e:x_e+W_e,:] = np.random.randint(0,255,(H_e,W_e,3))
                break

        return img

    def random_brightness(self, bgr):
        if random.random() < 0.5:
            return bgr

        hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        adjust = random.uniform(0.5, 1.5)
        v = v * adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv.merge((h, s, v))
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        return bgr
    
    def random_hue(self, bgr):
        if random.random() < 0.5:
            return bgr

        hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        adjust = random.uniform(0.8, 1.2)
        h = h * adjust
        h = np.clip(h, 0, 255).astype(hsv.dtype)
        hsv = cv.merge((h, s, v))
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        return bgr

    def random_saturation(self, bgr):
        if random.random() < 0.5:
            return bgr

        hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        adjust = random.uniform(0.5, 1.5)
        s = s * adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv.merge((h, s, v))
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        return bgr

    def rand_crop(self,img,boxes):
        '''
        对图片随机裁剪
        boxes:index,c,x,y,w,h
        '''
        h_orig,w_orig = img.shape[:2]
        h = np.random.uniform(h_orig*0.6,h_orig)  #裁剪过后的h
        w = np.random.uniform(w_orig*0.6,w_orig)  #w
        y = np.random.uniform(0,h_orig-h)  #裁剪过后的y坐标
        x = np.random.uniform(0,w_orig-w)  #x坐标

        #判断box是否还属于裁剪后的图像
        #xywh to xyxy
        xyxy = np.zeros_like(boxes[:,2:])
        xyxy[:,0] = (boxes[:,1]-boxes[:,3]/2) * w_orig
        xyxy[:,1] = (boxes[:,2]-boxes[:,4]/2) * h_orig
        xyxy[:,2] = (boxes[:,1]+boxes[:,3]/2) * w_orig  
        xyxy[:,3] = (boxes[:,2]+boxes[:,4]/2) * h_orig

        mask = (xyxy[:,0]>=x) & (xyxy[:,1]>=y) & (xyxy[:,2]<=x+w) & (xyxy[:,3]<=y+h)
        if mask.sum() == 0:
            return img,boxes

        return img[y:y+h,x:x+w,:],boxes[mask]

def collate_fn(batch):
    img,label = zip(*batch)
    for i,l in enumerate(label):
        l[:,0] = i
    return torch.stack(img,0),torch.cat(label,0)



def test():
    dataset = My_dataset('/Volumes/Elements/Algorithm/yolo/Yolov1/Data/VOC2007/train/images',448,True)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=collate_fn)
    for imgs,label in dataloader:
        img = imgs[0].numpy()
        img = img.transpose(1,2,0)
        print(label)
        cv_show(img)
        

if __name__ == '__main__':
    test()