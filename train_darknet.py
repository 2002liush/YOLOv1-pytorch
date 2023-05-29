import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from Model import *
from utils.yolo_utils import *

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torchvision

def update_lr(optimizer, epoch):
    if epoch == 0:
        lr = 0.01
    elif epoch == 45:
        lr = 0.001
    elif epoch == 85:
        lr = 0.0005
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(opt,device):
    save_dir,epoch,batch_size = \
        Path(opt.save_dir),opt.epoch,opt.batch_size


    save_dir.mkdir(parents=True,exist_ok=True)
    best_model_path = save_dir / 'best.pt'    #模型保存路径
    last_model_path = save_dir / 'last.pt'
    save_logs_path = save_dir / 'logs'         #tensorboard保存路径

    trans = []
    trans.append(torchvision.transforms.Resize(size=opt.img_size))
    trans.append(torchvision.transforms.ToTensor())


    transform = torchvision.transforms.Compose(trans)
    #构建dataset
    train = torchvision.datasets.CIFAR100(root='./Data/',train=True,transform=transform,
                                     download=True)
    test = torchvision.datasets.CIFAR100(root='./Data/',train=False,transform=transform,
                                     download=True)
    train_dataloader = DataLoader(dataset=train,batch_size=batch_size,shuffle=False)
    test_dataloader = DataLoader(dataset=test,batch_size=batch_size,shuffle=False)

    train_len = len(train)
    test_len = len(test)
    print('train data length:{}'.format(train_len))
    print('test data length:{}'.format(test_len))

    nc = 100
    model = Darknet(include_top=True,cls_num=nc).to(device)
    
    optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.00001)   #优化器
 
    loss_fn = nn.CrossEntropyLoss()
    
    writer = SummaryWriter(str(save_logs_path))

    total_loss = 0   #总损失    
    loss_res = []
    test_loss_res = []
    epoch_res = []
    acc_res = []
    max_acc = 0

    nb = len(train_dataloader)

    
    #开始训练
    for i in range(epoch):
        model.train()

        pbar = enumerate(train_dataloader)
        pbar = tqdm(pbar,total=nb)
        for _,(imgs,targets) in pbar:
            update_lr(optimizer,i)

            batch_size = imgs.shape[0]
            imgs = imgs.to(device).float() / 255.0

            pred = model(imgs)
            loss = loss_fn(pred,targets.to(device))
            total_loss += loss.cpu().detach().numpy() * batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        writer.add_scalar('loss',total_loss,i)
        print('epoch {}: train total loss:{}'.format(i+1,total_loss))
        
        loss_res.append(total_loss)
        epoch_res.append(i)
        total_loss = 0

        #对模型进行测试
        acc = 0
        model.eval()
        nb_test = len(test_dataloader)
        pbar_test = enumerate(test_dataloader)
        pbar_test = tqdm(pbar_test,total=nb_test) 
        with torch.no_grad():
            for _,(imgs,labels) in pbar_test:
                batch_size = imgs.shape[0]
                imgs = imgs.to(device).float() / 255.0
                pred = model(imgs)
                loss = loss_fn(pred,labels.to(device))
                total_loss += loss.cpu().detach().numpy() * batch_size
                acc += (pred.argmax(1) == labels.to(device)).sum().item()

            acc = acc / test_len
            acc_res.append(acc)
            test_loss_res.append(total_loss)
            if acc > max_acc:
                torch.save(model.state_dict(),best_model_path)
                max_acc = acc
            print('epoch {}: test total loss:{}  accuracy:{}\n'.format(i+1,total_loss,acc))

            total_loss = 0

        torch.save(model.state_dict(),last_model_path)

    fig,ax = plt.subplots(1,2,figsize = (8,4))
    ax[0].plot(epoch_res,loss_res)
    ax[0].plot(epoch_res,test_loss_res)
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')

    ax[1].plot(epoch_res,acc_res)
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('acc')
    
    fig.savefig(str(save_dir)+'/res.png')


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',type=str,default='best.pt',help='initial weights path')
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--epoch',type=int,default=100)
    parser.add_argument('--img_size',type=int,default=(224,224))
    parser.add_argument('--name',type=str,default='darknet_exp',help='save to project/name')
    parser.add_argument('--project',type=str,default='run/train')
    parser.add_argument('--device',type=str,default='0',help='cpu or 0 or 0,1,2,3')

    opt = parser.parse_args()
    device = select_device(opt.device)

    opt.save_dir = increment_path(Path(opt.project) / opt.name,exist_ok=False)
    print('Device:{}'.format(device))
    print('result has written to {}'.format(opt.save_dir))
    train(opt,device)



