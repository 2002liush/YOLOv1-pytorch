from time import sleep
import torch
import torch.nn as nn
from utils.yolo_utils import cal_IOU
from torch.autograd import Variable
import torch.nn.functional as F

class Yolov1_Loss(nn.Module):
    def __init__(self,img_size) -> None:
        super().__init__()
        
        self.img_size = img_size
    def forward(self,output,labels,device):
        '''
        pred:[batch_size,7,7,30],模型的预测
        labels:[nl,6],标签,nl代表所有标签数
        '''
        batch_size = int(output.shape[0])

        coor_loss = 0  #坐标损失，只针对含有目标的预测框
        conf_loss = 0   #置信度损失
        class_loss = 0 #对含有目标的方格的类别损失

        w_coord = 5
        w_noobj = 0.5

        for batch in range(batch_size):
            bool_label = labels[:,0]==batch
            label_one_img = labels[bool_label,:]

            conf_loss += w_noobj * torch.sum(output[batch,:,:,20]**2) + w_noobj*torch.sum(output[batch,:,:,21]**2)

            if label_one_img is not None:
                for l in label_one_img:   #l为1*6的数组
                    #计算当前这个label属于图片当中的第几个grid cell
                    row = int(l[2]*7)
                    col = int(l[3]*7)

                    #计算含有目标的类别损失
                    label_class = torch.zeros(20).to(device)
                    label_class[int(l[1])] = 1  #l[1]为当前标签所属的类别
                    class_loss += torch.sum((output[batch,row,col,:20]-label_class)**2)
                    
                
                    box1 = torch.clone(output[batch,row,col,22:26])
                    box2 = torch.clone(output[batch,row,col,26:30])
                    box1[0] = (row * self.img_size / 7 + self.img_size / 7 * box1[0]) / self.img_size
                    box1[1] = (col * self.img_size / 7 + self.img_size / 7 * box1[1]) / self.img_size
                    box2[0] = (row * self.img_size / 7 + self.img_size / 7 * box2[0]) / self.img_size
                    box2[1] = (col * self.img_size / 7 + self.img_size / 7 * box2[1]) / self.img_size
                    
                    #计算方框与预测框之间的iou
                    iou_1 = cal_IOU(box1,l[2:6])
                    iou_2 = cal_IOU(box2,l[2:6])
                    gt = torch.clone(l[2:])

                    gt[0] = (gt[0]*self.img_size - self.img_size/7*row)/self.img_size*7
                    gt[1] = (gt[1]*self.img_size - self.img_size/7*col)/self.img_size*7
                    
                    e = 1e-6
                    if(iou_1>=iou_2):
                        
                        coor_loss += w_coord * torch.sum((output[batch,row,col,22:24]-gt[:2])**2)
                        coor_loss += w_coord * torch.sum((torch.sqrt(output[batch,row,col,24:26]+e)-torch.sqrt(gt[2:4]+e))**2)
                        conf = iou_1
                        conf_loss += torch.sum((output[batch,row,col,20] - conf)**2) - w_noobj*torch.sum(output[batch,row,col,20]**2)

                    else:
                        coor_loss += w_coord * torch.sum((output[batch,row,col,26:28]-gt[:2])**2)
                        coor_loss += w_coord * torch.sum((torch.sqrt(output[batch,row,col,28:30]+e)-torch.sqrt(gt[2:4]+e))**2)
                        conf = iou_2
                        conf_loss += torch.sum((output[batch,row,col,21] - conf)**2) - w_noobj*torch.sum(output[batch,row,col,21]**2)


        total_loss =  (coor_loss + conf_loss + class_loss) / float(batch_size)

        return total_loss

class Loss(nn.Module):

    def __init__(self, feature_size=7, num_bboxes=2, num_classes=20, lambda_coord=5.0, lambda_noobj=0.5):
        """ Constructor.
        Args:
            feature_size: (int) size of input feature map.
            num_bboxes: (int) number of bboxes per each cell.
            num_classes: (int) number of the object classes.
            lambda_coord: (float) weight for bbox location/size losses.
            lambda_noobj: (float) weight for no-objectness loss.
        """
        super(Loss, self).__init__()

        self.S = feature_size
        self.B = num_bboxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj


    def compute_iou(self, bbox1, bbox2):
        """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
        Args:
            bbox1: (Tensor) bounding bboxes, sized [N, 4].
            bbox2: (Tensor) bounding bboxes, sized [M, 4].
        Returns:
            (Tensor) IoU, sized [N, M].
        """
        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt   # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0 # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter # [N, M, 2]
        iou = inter / union           # [N, M, 2]

        return iou

    def forward(self, pred_tensor, target_tensor):
        """ Compute loss for YOLO training.
        Args:
            pred_tensor: (Tensor) predictions, sized [n_batch, S, S, Bx5+C], 5=len([x, y, w, h, conf]).
            target_tensor: (Tensor) targets, sized [n_batch, S, S, Bx5+C].
        Returns:
            (Tensor): loss, sized [1, ].
        """
        # TODO: Romove redundant dimensions for some Tensors.

        S, B, C = self.S, self.B, self.C
        N = 5 * B + C    # 5=len([x, y, w, h, conf]

        batch_size = pred_tensor.size(0)
        coord_mask = target_tensor[:, :, :, 4] > 0  # mask for the cells which contain objects. [n_batch, S, S]
        noobj_mask = target_tensor[:, :, :, 4] == 0 # mask for the cells which do not contain objects. [n_batch, S, S]
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor) # [n_batch, S, S] -> [n_batch, S, S, N]
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor) # [n_batch, S, S] -> [n_batch, S, S, N]

        coord_pred = pred_tensor[coord_mask].view(-1, N)            # pred tensor on the cells which contain objects. [n_coord, N]
                                                                    # n_coord: number of the cells which contain objects.
        bbox_pred = coord_pred[:, :5*B].contiguous().view(-1, 5)    # [n_coord x B, 5=len([x, y, w, h, conf])]
        class_pred = coord_pred[:, 5*B:]                            # [n_coord, C]

        coord_target = target_tensor[coord_mask].view(-1, N)        # target tensor on the cells which contain objects. [n_coord, N]
                                                                    # n_coord: number of the cells which contain objects.
        bbox_target = coord_target[:, :5*B].contiguous().view(-1, 5)# [n_coord x B, 5=len([x, y, w, h, conf])]
        class_target = coord_target[:, 5*B:]                        # [n_coord, C]

        # Compute loss for the cells with no object bbox.
        noobj_pred = pred_tensor[noobj_mask].view(-1, N)        # pred tensor on the cells which do not contain objects. [n_noobj, N]
                                                                # n_noobj: number of the cells which do not contain objects.
        noobj_target = target_tensor[noobj_mask].view(-1, N)    # target tensor on the cells which do not contain objects. [n_noobj, N]
                                                                # n_noobj: number of the cells which do not contain objects.
        noobj_conf_mask = torch.cuda.ByteTensor(noobj_pred.size()).fill_(0) # [n_noobj, N]
        for b in range(B):
            noobj_conf_mask[:, 4 + b*5] = 1 # noobj_conf_mask[:, 4] = 1; noobj_conf_mask[:, 9] = 1
        noobj_pred_conf = noobj_pred[noobj_conf_mask]       # [n_noobj, 2=len([conf1, conf2])]
        noobj_target_conf = noobj_target[noobj_conf_mask]   # [n_noobj, 2=len([conf1, conf2])]
        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')
        
        # Compute loss for the cells with objects.
        coord_response_mask = torch.cuda.ByteTensor(bbox_target.size()).fill_(0)    # [n_coord x B, 5]
        coord_not_response_mask = torch.cuda.ByteTensor(bbox_target.size()).fill_(1)# [n_coord x B, 5]
        bbox_target_iou = torch.zeros(bbox_target.size()).cuda()                    # [n_coord x B, 5], only the last 1=(conf,) is used
        
        # Choose the predicted bbox having the highest IoU for each target bbox.
        for i in range(0, bbox_target.size(0), B):
            pred = bbox_pred[i:i+B] # predicted bboxes at i-th cell, [B, 5=len([x, y, w, h, conf])]
            pred_xyxy = Variable(torch.FloatTensor(pred.size())) # [B, 5=len([x1, y1, x2, y2, conf])]
            # Because (center_x,center_y)=pred[:, 2] and (w,h)=pred[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            pred_xyxy[:,  :2] = pred[:, :2]/float(S) - 0.5 * pred[:, 2:4]
            pred_xyxy[:, 2:4] = pred[:, :2]/float(S) + 0.5 * pred[:, 2:4]

            target = bbox_target[i] # target bbox at i-th cell. Because target boxes contained by each cell are identical in current implementation, enough to extract the first one.
            target = bbox_target[i].view(-1, 5) # target bbox at i-th cell, [1, 5=len([x, y, w, h, conf])]
            target_xyxy = Variable(torch.FloatTensor(target.size())) # [1, 5=len([x1, y1, x2, y2, conf])]
            # Because (center_x,center_y)=target[:, 2] and (w,h)=target[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            target_xyxy[:,  :2] = target[:, :2]/float(S) - 0.5 * target[:, 2:4]
            target_xyxy[:, 2:4] = target[:, :2]/float(S) + 0.5 * target[:, 2:4]

            iou = self.compute_iou(pred_xyxy[:, :4], target_xyxy[:, :4]) # [B, 1]
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()

            coord_response_mask[i+max_index] = 1
            coord_not_response_mask[i+max_index] = 0

            # "we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth"
            # from the original paper of YOLO.
            bbox_target_iou[i+max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        bbox_target_iou = Variable(bbox_target_iou).cuda()

        # BBox location/size and objectness loss for the response bboxes.
        bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 5)      # [n_response, 5]
        bbox_target_response = bbox_target[coord_response_mask].view(-1, 5)  # [n_response, 5], only the first 4=(x, y, w, h) are used
        target_iou = bbox_target_iou[coord_response_mask].view(-1, 5)        # [n_response, 5], only the last 1=(conf,) is used
        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_target_response[:, 2:4]), reduction='sum')
        loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')

        # Class probability loss for the cells which contain objects.
        loss_class = F.mse_loss(class_pred, class_target, reduction='sum')
        
        # Total loss
        loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_noobj + loss_class
        loss = loss / float(batch_size)

        return loss

class yolo_Loss(nn.Module):
    def __init__(self,S = 7,B = 2,cls_num = 20,lambda_coord = 5,lambda_noobj = 0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.class_num = cls_num
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def compute_iou(self, bbox1, bbox2):
        """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
        Args:
            bbox1: (Tensor) bounding bboxes, sized [N, 4].
            bbox2: (Tensor) bounding bboxes, sized [M, 4].
        Returns:
            (Tensor) IoU, sized [N, M].
        """
        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt   # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0 # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter # [N, M, 2]
        iou = inter / union           # [N, M, 2]

        return iou


    def forward(self,pred,target):
        '''
        pred:[batch,S,S,N]
        target:[batch,S,S,N]
        '''
        N = self.class_num + self.B * 5
        cls_num,B,S = self.class_num,self.B,self.S
        batch_size = pred.shape[0]
        coord_mask = target[:,:,:,4] == 1  #[batch,S,S]
        noobj_mask = target[:,:,:,4] == 0  #[batch,S,S]

        pred_coord = pred[coord_mask].view(-1,N) #[n,N]
        pred_noobj = pred[noobj_mask].view(-1,N) #[m,N]
        target_coord = target[coord_mask].view(-1,N) #[n,N]
        target_noobj = target[noobj_mask].view(-1,N) #[m,N]

        #compute loss for noobj
        conf_loc = torch.arange(0,self.B)*self.B + 4
        target_noobj_conf = target_noobj[:,conf_loc]
        pred_noobj_conf = pred_noobj[:,conf_loc]
        noobj_conf_loss = F.mse_loss(pred_noobj_conf,target_noobj_conf,reduction='sum')

        #compute loss for coord
        pred_boxes = pred_coord[:,:B*5].view(-1,5)   #[2n,5]
        pred_class = pred_coord[:,B*5:]              #[n,20]
        target_boxes = target_coord[:,:5]            #[n,5]
        target_class = target_coord[:,B*5:]          #[n,20]

        

        for i in range(0,target_coord.shape[0]):
            box = pred_boxes[i:i+B]             #[2,5]
            target_box = target_boxes[i].unsqueeze(0)   #[1,5]
            pred_box_xyxy = torch.zeros_like(box)                              #[x,y,w,h] to [x,y,x,y]
            pred_box_xyxy[:,:2] = box[:,:2] - box[:,2:] / 2
            pred_box_xyxy[:,2:] = box[:,:2] + box[:,2:] / 2

            target_box_xyxy = torch.zeros_like(target_box)                              #[x,y,w,h] to [x,y,x,y]
            target_box_xyxy[:,:2] = target_box[:,:2] - target_box[:,2:] / 2
            target_box_xyxy[:,2:] = target_box[:,:2] + target_box[:,2:] / 2

            box_iou = self.compute_iou(pred_box_xyxy,target_box_xyxy)

            max_iou,max_index = box_iou.max(0)



# loss = Yolov1_Loss()

# output = torch.ones((10,7,7,30))
# target = torch.ones((10,7,7,30))
# loss_fn = loss(output,target)

# print(loss_fn)







