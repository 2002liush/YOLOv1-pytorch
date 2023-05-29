import argparse
from pyexpat import model
import time
from tqdm import tqdm
from pathlib import Path
import numpy as np
import yaml

from Model import *

from utils.yolo_utils import ap_per_class, box_iou,increment_path, nms, select_device
from utils.datasets import My_dataset, collate_fn

from torch.utils.data import DataLoader
import torch


def test_VOC(test_path,
            save_dir,
            device,
            class_name,
            model,
            img_size = 448,
            batch_size = 64,
            conf_thresh_start = 0.001,
            iou_thresh = 0.6):
    
    save_dir.mkdir(parents=True,exist_ok=True)

    #Dataloader
    dataset = My_dataset(test_path,img_size)
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_fn)

    nb = len(dataloader)
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar,total=nb)
    
    state = []
    iouv = np.linspace(0.5,0.95,10)
    niou = len(iouv)

    #训练模式
    model.eval()
    
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar,total=nb)

    for _,(imgs,targets) in pbar:
        
        t0 = time.time()
        batch_size = imgs.shape[0]
        #预测
        imgs = imgs.to(device).float() / 255.0
        with torch.no_grad():
            pred = model(imgs)

        # pred = pred.cpu().detach().numpy()
        #将预测的x和y转化为相对于整个图像的x,y
        for row in range(7):
            for col in range(7):
                pred[:,col,row,0] = ((img_size / 7)*pred[:,row,col,0] + row*img_size/7) / img_size
                pred[:,col,row,1] = ((img_size / 7)*pred[:,row,col,1] + col*img_size/7) / img_size
                pred[:,col,row,5] = ((img_size / 7)*pred[:,row,col,5] + row*img_size/7) / img_size
                pred[:,col,row,6] = ((img_size / 7)*pred[:,row,col,6] + col*img_size/7) / img_size

        pred = pred.reshape((batch_size,-1,30))

        pred = pred.cpu().detach().numpy()
        t1 = time.time()
        print('shift:{}'.format(t1-t0))
        for i in range(batch_size):
            t0 = time.time()
            #将一张图片中的预测框全部提取出来
            pred_one_img = pred[i]   #49 * 30
            dts = []
            for l in pred_one_img:
                cls = np.argmax(l[10:])
                dts.append([cls,l[4],l[0],l[1],l[2],l[3]])  #添加所有预测框
                dts.append([cls,l[9],l[5],l[6],l[7],l[8]])

            dts = np.array(dts)  #转化为ndarry
            

            t1 = time.time()
            print('before nms:{}'.format(t1-t0))
            #对预测框进行nms 
            detect = nms(dts,iou_thresh,conf_thresh_start)
            
            label = targets[targets[:,0] == i,1:]  #属于当前图片的label

            nl = len(label)   #number of label
            tcls = label[:,0].tolist() if nl else []

            if len(detect) == 0:
                if nl:
                    state.append((np.zeros(0,niou,dtype=bool),np.array(),np.array(),tcls))

            correct = np.zeros((detect.shape[0],niou),dtype=bool,device = device)
            if nl:
                detected = []
                for cls in np.unique(tcls):
                    pi = (cls == detect[:,0]).nonzero()[0]   #prediction indices
                    ti = (cls == np.array(tcls)).nonzero()[0]

                    if pi.shape[0]:
                        iou,i = box_iou(detect[pi,2:],label[ti,1:]).max(1)

                        detceted_set = set()
                        for j in (iou>iouv[0]).nonzero():
                            d = ti[i[j]]
                            if d not in detceted_set:
                                correct[pi[j]] = iou[j].numpy() > iouv
                                detceted_set.add(d)
                                detected.append(d)
                                if len(detected) == nl:
                                    break
            state.append((correct,detect[:,1],detect[:,0],tcls))

    state = [np.concatenate(x,0) for x in zip(*state)]

    ap = ap_per_class(state[0],state[1],state[2],state[3],save_dir,class_name)
        
    ap50,ap = ap[:,0],ap.mean(1)

    map50,map= ap50.mean(),ap.mean()

    return map50,map


                
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',type=str,default='./last.pt',help='initial weights path')
    parser.add_argument('--data',type=str,default='./Data/voc.yaml',help='data cfg file path')
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--conf_thresh',type=float,default=0.001)
    parser.add_argument('--iou_thresh',type=float,default=0.6,help='for nms')
    parser.add_argument('--img_size',type=int,default=448)
    parser.add_argument('--name',type=str,default='exp',help='save to project/name')
    parser.add_argument('--project',type=str,default='run/test')
    parser.add_argument('--device',type=str,default='0',help='cpu or 0 or 0,1,2,3')

    opt = parser.parse_args()

    device = select_device(opt.device)

    opt.save_dir = increment_path(Path(opt.project) / opt.name,False)
    print('Device:{}'.format(device))
    print('result has written to {}'.format(opt.save_dir))

    with open(opt.data) as f:      #加载数据文件
        data_dict = yaml.load(f,Loader=yaml.SafeLoader)

    class_name = data_dict['names']   #训练集路径
    test_path = data_dict['val']      #测试集路径
    # darknet = Darknet().to(device)
    # model = yolov1(darknet.feature).to(device)
    # model.load_state_dict(torch.load(opt.weights))
    model = yolov1_ResNet().to(device)
    model.load_state_dict(torch.load(opt.weights,map_location=torch.device('cpu')))
    map50,map = test_VOC(test_path=test_path,
                            device = device,
                            save_dir=Path(opt.save_dir),
                            class_name=class_name,
                            model = model,
                            batch_size=opt.batch_size)
    print(map50,map)