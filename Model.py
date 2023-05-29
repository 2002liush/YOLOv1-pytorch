import torch
import torch.nn as nn
import torchvision.models as models
#import torchinfo         
class Block(nn.Module):

    def expansion(self):
        return 2

    def __init__(self,in_channel,out_channel,stride = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel,out_channel*self.expansion(),kernel_size=3,stride=1,padding='same')
        self.bn2 = nn.BatchNorm2d(out_channel*self.expansion())

        self.relu = nn.LeakyReLU(0.1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class Conv(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride = 1,pad = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size,stride,pad)#448->224
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(0.1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()
        
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Darknet(nn.Module):
    def __init__(self,include_top = False,cls_num = 1000):
        super().__init__()
        
        self.include_top = include_top
        self.cls_num = cls_num
        self.feature = self._make_conv_bn_layers()
        if self.include_top:
            self.fc = self._make_fc_layers()

        self._initialize_weights()

    def forward(self,x):
        x = self.feature(x)
        if self.include_top:
            x = self.fc(x)

        return x

    def _make_conv_bn_layers(self): 
        conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(192, 128, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        return conv 
    
    def _make_fc_layers(self):
        fc = nn.Sequential(
            nn.AvgPool2d(7),
            Squeeze(),
            nn.Linear(1024, self.cls_num)
        )
        return fc

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class yolov1(nn.Module):
    def __init__(self,feature,cls_num = 20):
        super().__init__()
        self.features = feature
        
        self.conv = self._make_conv_layers(bn=True)
        self.fc = self._make_fc_layers()


    def forward(self,x):
        x = self.features(x)
        x = self.conv(x)
        x = self.fc(x)

        x = x.view(-1,7,7,30)

        return x

    def _make_conv_layers(self, bn):
        if bn:
            net = nn.Sequential(
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),

                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True)
            )

        else:
            net = nn.Sequential(
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),

                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True)
            )

        return net

    def _make_fc_layers(self):
        S, B, C = 7,2,20

        net = nn.Sequential(
            Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5, inplace=False), # is it okay to use Dropout with BatchNorm?
            nn.Linear(4096, S * S * (5 * B + C)),
            nn.Sigmoid()
        )

        return net



class yolov1_ResNet(nn.Module):
    def __init__(self,cls_num = 20) -> None:
        super().__init__()

        resnet50 = models.resnet34(pretrained=True)
        self.new_model = torch.nn.Sequential(*( list(resnet50.children())[:-2]))
        self.conv1 = Conv(512,1024,kernel_size=3,stride=1,pad=1)
        self.conv2 = Conv(1024,1024,kernel_size=3,stride=2,pad=1)
        self.conv3 = Conv(1024,1024,kernel_size=3,stride=1,pad=1)
        self.conv4 = Conv(1024,1024,kernel_size=3,stride=1,pad=1)

        self.classfifier = nn.Sequential(
            nn.Linear(1024*7*7,4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Linear(4096,7*7*30),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.new_model(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.flatten(x,1)
        x = self.classfifier(x)

        x = torch.reshape(x,(x.shape[0],7,7,30))
        
        return x

def test_darknet():
    input = torch.zeros((10,3,224,224))
    model = Darknet(True)
    output = model(input)

    print(output.shape)

def test_yolov1():
    input = torch.zeros((10,3,448,448))
    darknet = Darknet(True)
    yolo = yolov1(darknet.feature)

    output = yolo(input)
    print(output.shape)

# model = yolov1_2()
#new_model = torch.nn.Sequential( *( list(model.children())[:-2] ) )
# x = torch.zeros((4,3,448,448))
# output = model(x)
# print(output.shape)
# print(list(model.children()))
#print(torchinfo.summary(model, (3, 224, 224), batch_dim = 0, col_names = ('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'), verbose = 0))
if __name__ == '__main__':
    # test_yolov1()
    input = torch.zeros((4,3,448,448))
    vgg16 = models.vgg16()
    new_model = torch.nn.Sequential(*( list(vgg16.children())[:-2]))
    output = new_model(input)
    #print(list(new_model.children()))
    print(output.shape)