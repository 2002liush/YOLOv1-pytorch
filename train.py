import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import math

from utils.Loss import Yolov1_Loss
from Model import *
from utils.datasets import My_dataset, collate_fn
from utils.yolo_utils import increment_path, select_device

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader

# Training hyper parameters.
init_lr = 0.001
base_lr = 0.01

def update_lr(optimizer, epoch, burnin_base, burnin_exp=4.0):
    if epoch == 0:
        lr = init_lr + (base_lr - init_lr) * math.pow(burnin_base, burnin_exp)
    elif epoch == 1:
        lr = 0.01
    elif epoch == 75:
        lr = 0.001
    elif epoch == 105:
        lr = 0.0001
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(opt,device):
    save_dir,batch_size = \
        Path(opt.save_dir),opt.batch_size


    save_dir.mkdir(parents=True,exist_ok=True)
    best_path = save_dir / 'best.pt'    #模型保存路径
    last_path = save_dir / 'last.pt'
    save_logs_path = save_dir / 'logs'         #tensorboard保存路径
    result_file = save_dir / 'result.txt'

    with open(opt.data) as f:      #加载数据文件
        data_dict = yaml.load(f,Loader=yaml.SafeLoader)

    train_path = data_dict['train']   #训练集路径
    test_path = data_dict['val']      #测试集路径
    class_name = data_dict['names']   #类别名称

    #train
    dataset = My_dataset(train_path,opt.img_size[0],augmentation=True)
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_fn,num_workers=4)
    #test
    test_dataset = My_dataset(test_path,opt.img_size[0])
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_fn,num_workers=4)

    train_len = len(dataset)
    test_len = len(test_dataset)
    print('train image:{}'.format(train_len))
    print('test image:{}'.format(test_len))


    #model = torch.load(opt.weights,map_location=torch.device(device))  #加载模型  
    #model = yolov1(20).to(device)
    #model.load_state_dict(torch.load(opt.weights))
    checkpoint = './darknet.pt'
    darknet = Darknet(include_top=True,cls_num=100).to(device)
    darknet.load_state_dict(torch.load(checkpoint))

    model = yolov1(darknet.feature).to(device)
    
    optimizer = optim.SGD(model.parameters(),lr=init_lr,momentum=0.9,weight_decay=0.0005)   #优化器
    #optimizer = optim.Adam(model.parameters(),lr=0.01,weight_decay=0.0005)

    loss_fn = Yolov1_Loss(opt.img_size[0])   #自定义损失函数
 
    #scaler = amp.GradScaler(enabled=cuda)

    writer = SummaryWriter(str(save_logs_path))

    total_loss = 0   #总损失    
    loss_res = []
    test_loss_res = []
    epoch_res = []

    nb = len(dataloader)

    best_loss = 1e+16

    f = open(result_file,'w')
    for i in range(opt.epoch):
        model.train()

        pbar = enumerate(dataloader)
        pbar = tqdm(pbar,total=nb)

        for j,(imgs,targets) in pbar:
            batch_size = imgs.shape[0]

            update_lr(optimizer,i,float(j) / float(nb - 1))

            imgs = imgs.to(device).float() / 255.0
            
            pred = model(imgs)
            loss = loss_fn(pred,targets.to(device),device)
            #print(loss)
            total_loss += loss.cpu().detach().numpy() * batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        torch.save(model.state_dict(),last_path)  #保存最新的模型

        writer.add_scalar('loss',total_loss,i)
        s = ''
        s += 'epoch {}: total loss:{}  '.format(i+1,total_loss)
        print('epoch {}: total loss:{}'.format(i+1,total_loss))
        
        loss_res.append(total_loss)
        epoch_res.append(i)
        total_loss = 0

        model.eval()
        pbar = enumerate(test_dataloader)
        pbar = tqdm(pbar,total=len(test_dataloader))
        for _,(imgs,targets) in pbar:
            batch_size = imgs.shape[0]
            imgs = imgs.to(device).float() / 255.0
            with torch.no_grad():
                pred = model(imgs)
            loss = loss_fn(pred,targets.to(device),device)
            #print(loss)
            total_loss += loss.cpu().detach().numpy() * batch_size

        if total_loss < best_loss:
            torch.save(model.state_dict(),best_path)
            best_loss = total_loss

        test_loss_res.append(total_loss)
        
        s += 'test total loss:{}\n'.format(total_loss)
        print('epoch {}: test total loss:{}\n'.format(i+1,total_loss))
        f.write(s)
        total_loss = 0

    # test_model = yolov1_ResNet().to(device)
    # test_model.load_state_dict(torch.load(best_path))
    
    # map50,map = test_VOC(test_path,save_dir,device,class_name,model=test_model)
    # s = 'map@.5:' + str(map50)[:4] + ' map:' + str(map)[:4]
    #torch.save(model.state_dict(),last_path)  #保存最新的模型

    fig = plt.figure(num = 0,figsize=(5,5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(epoch_res,loss_res)
    ax.plot(epoch_res,test_loss_res)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')

    fig.savefig(str(save_dir)+'/loss_res.png')

    f.close()


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',type=str,default='run/train/exp6/best.pt',help='initial weights path')
    parser.add_argument('--data',type=str,default='./Data/voc.yaml',help='data cfg file path')
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--epoch',type=int,default=150)
    parser.add_argument('--img_size',type=int,default=[448,448])
    parser.add_argument('--name',type=str,default='exp',help='save to project/name')
    parser.add_argument('--project',type=str,default='run/train')
    parser.add_argument('--device',type=str,default='0',help='cpu or 0 or 0,1,2,3')

    opt = parser.parse_args()
    device = select_device(opt.device)

    opt.save_dir = increment_path(Path(opt.project) / opt.name,False)
    print('Device:{}'.format(device))
    print('result has written to {}'.format(opt.save_dir))
    train(opt,device)



