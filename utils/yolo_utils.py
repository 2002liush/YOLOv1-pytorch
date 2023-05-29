import time
import torch
import glob
import numpy as np
import os
from pathlib import Path
import re
import matplotlib.pyplot as plt

def select_device(device = ''):
    cpu = device.lower() == 'cpu'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    cuda = not cpu and torch.cuda.is_available()

    return torch.device('cuda:0' if cuda else 'cpu')

def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def xywh2xyxy(x):
    y = [0,0,0,0]
    y[0] = max(x[0] - x[2] / 2,0)
    y[1] = max(x[1] - x[3] / 2,0)
    y[2] = min(x[0] + x[2] / 2,1)
    y[3] = min(x[1] + x[3] / 2,1)

    return y

#iou为两框交集除以并集
def cal_IOU(pred,label):
    '''
    pred表示预测出的预测框的位置,为x,y,w,h
    label为真实框的位置
    '''
    S1 = pred[2] * pred[3]
    S2 = label[2] * label[3]
    box1 = xywh2xyxy(pred)
    box2 = xywh2xyxy(label)

    x_1_left,x_1_right,y_1_left,y_1_right = box1[0],box1[2],box1[1],box1[3]
    x_2_left,x_2_right,y_2_left,y_2_right = box2[0],box2[2],box2[1],box2[3]

    x_max = max(x_1_left,x_2_left)
    y_max = max(y_1_left,y_2_left)
    x_min = min(x_1_right,x_2_right)
    y_min = min(y_1_right,y_2_right)

    inter_w = max(x_min - x_max,0)
    inter_h = max(y_min - y_max,0)

    inter = inter_h * inter_w
    union = S1 + S2 - inter

    return inter / union

    
#非极大值抑制
def nms(dt,iou_thresh,conf_thresh):
    #l:conf_class,conf_frame,x,y,w,h
    # t0 = time.time()
    dt = dt[dt[...,1] >= conf_thresh]   #根据置信度阈值删除一部分框
    
    cls_unique = np.unique(dt[:,0])    #预测框所包含的所有类别
    
    res = []
    for cls in cls_unique:     #对每一类框进行nms
        keep = []
        boxes = dt[dt[:,0] == cls]
        i = np.argsort(-boxes[:,1]) 
        boxes = boxes[i]             #将预测框置信度从大到小排列
        if boxes.shape[0] == 1:
            res.append(boxes[0])
            continue
        while True:
            iou = box_iou(boxes[1:,2:],[boxes[0,2:]])
            loc = (iou > iou_thresh).nonzero()
            res.append(boxes[0])
            boxes = np.delete(boxes[1:],loc,axis=0)
            if boxes.shape[0] == 1:
                res.append(boxes[0])
                break
            elif boxes.shape[0] == 0:
                break
        
    res = np.vstack(res)

    # t1 = time.time()
    # print('nms:{}'.format(t1-t0))
    return res

def box_iou(box1,box2):
    '''
    box1:N*4 [x,y,w,h]
    box1:M*4 [x,y,w,h]
    return: [N,M]
    '''
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    t0 = time.time()
    box1 = [xywh2xyxy(x) for x in box1]
    box2 = [xywh2xyxy(x) for x in box2]
    box1 = torch.tensor(box1)
    box2 = torch.tensor(box2)

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:,None,2:],box2[:,2:]) - torch.max(box1[:,None,:2],box2[:,:2])).clamp(0).prod(2)

    t1 = time.time()
    print('iou time:{}'.format(t1-t0))
    return inter / (area1[:,None] + area2 - inter)

def ap_per_class(pred,conf,pred_cls,target_cls,save_dir,name):
    
    name = np.array(name)
    loc = np.argsort(-conf)
    pred = pred[loc]
    conf = conf[loc]
    pred_cls = pred_cls[loc]

    unique_class = np.unique(target_cls)
    nc = len(unique_class)

    px,py= np.linspace(0,1,1000),[]
    p,r,ap = np.zeros((nc,1000)),np.zeros((nc,1000)),np.zeros((nc,pred.shape[1]))
    #对每个类计算p，r，ap
    for ci,cls in enumerate(unique_class):
        i = pred_cls == cls

        n_l = (target_cls == cls).sum()  #number of label
        n_p = i.sum()                    #number of prediction

        if n_l == 0 or n_p == 0:
            continue
        
        else:
            fp = (1 - pred[i]).cumsum(0)
            tp = pred[i].cumsum(0)

            #Recall
            recall = tp / (n_l + 1e-16)

            r[ci] = np.interp(-px,-conf[i],recall[:,0],left=0)
            #precision
            precision = tp / (tp + fp)
            p[ci] = np.interp(-px,-conf[i],precision[:,0],left=1)

            for j in range(pred.shape[1]):
                ap[ci,j],ap_5_r,ap_5_p= compute_ap(recall[:,j],precision[:,j])
                if j==0:
                    py.append(np.interp(px,ap_5_r,ap_5_p))


    plot_PR_curve(px,py,ap,save_dir / 'PR_curve.jpg',name[np.array(unique_class,dtype = int)])

    return ap    

def compute_ap(recall,precision):

    r = np.concatenate(([0.0],recall,[1.0]))
    p = np.concatenate(([1.0],precision,[0.0]))

    p = np.flip(np.maximum.accumulate(np.flip(p)))

    x = np.linspace(0,1,101)   #横坐标101个点
    ap = np.trapz(np.interp(x,r,p),x)

    return ap,r,p


def plot_PR_curve(px,py,ap,save_dir,names):

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py,axis = 1)

    for i,y in enumerate(py.T):
        ax.plot(px,y,linewidth = 1,label = f'{names[i]} {ap[i,0]:.3f}')

    ax.plot(px,py.mean(1),linewidth = 3,label = 'all classes %.3f mAP@.5' % ap[:,0].mean())
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir),dpi = 250)

# a = np.array([0.25,0.25,0.2,0.2])
# b = np.array([0.75,0.25,0.2,0.2])
# print(cal_IOU(a,b))

    
