import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import time

#******************************  VGGNET16 ******************************#
class VggNet16(nn.Module):
    def __init__(self):
        super(VggNet16,self).__init__()
        self.convnet=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1,stride=1),nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,stride=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1,stride=1),nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1,stride=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(128,256,3,padding=1,stride=1),nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1,stride=1),nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1,stride=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        
            nn.Conv2d(256,512,3,padding=1,stride=1),nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1,stride=1),nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1,stride=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(512,512,3,padding=1,stride=1),nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1,stride=1),nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1,stride=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
            )

    # The original vggnet used Linear(512*N*N,4096), Linear(4096,4096), 
    # but since our dataset is small, we modified the Linear layer
        self.fclayer=nn.Sequential(
            nn.Linear(512*1*1,200),
            nn.ReLU(),
            nn.Linear(200,200),
            nn.ReLU(),
            nn.Linear(200,num_classes)
            )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

    def forward(self,x):
        x=self.convnet(x)
        x=x.view(-1,512*1*1)
        x=self.fclayer(x)
        return x

#****************************** VGGNET19 ******************************#
class VggNet19(nn.Module):
    def __init__(self):
        super(VggNet19,self).__init__()
        self.convnet=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1,stride=1),nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,stride=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1,stride=1),nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1,stride=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(128,256,3,padding=1,stride=1),nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1,stride=1),nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1,stride=1),nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1,stride=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(256,512,3,padding=1,stride=1),nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1,stride=1),nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1,stride=1),nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1,stride=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(512,512,3,padding=1,stride=1),nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1,stride=1),nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1,stride=1),nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1,stride=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
            )

        self.fclayer=nn.Sequential(
            nn.Linear(512*1*1,200),
            nn.ReLU(),
            nn.Linear(200,200),
            nn.ReLU(),
            nn.Linear(200,num_classes)
            )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

    def forward(self,x):
        x=self.convnet(x)
        x=x.view(-1,512*1*1)
        x=self.fclayer(x)
        return x



#******************************  RESNET18,34,50  ******************************#
class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()
        self.residual_function=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels*BasicBlock.expansion,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels*BasicBlock.expansion)
            )

        self.shortcut=nn.Sequential()

        self.relu=nn.ReLU()

        if stride!=1 or in_channels!=BasicBlock.expansion*out_channels:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels,out_channels*BasicBlock.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels*BasicBlock.expansion)
                )

    def forward(self,x):
        x=self.residual_function(x)+self.shortcut(x)
        x=self.relu(x)
        return x


class BottleNeck(nn.Module):
    expansion=4
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()

        self.residual_function=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels*BottleNeck.expansion,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )

        self.shortcut=nn.Sequential()
        self.relu=nn.ReLU()

        if stride!=1 or in_channels != out_channels*BottleNeck.expansion:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels,out_channels*BottleNeck.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
                )

    def forward(self,x):
        x=self.residual_function(x)+self.shortcut(x)
        x=self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self,block, num_block,num_classes=50,init_weights=True):
        super().__init__()

        self.in_channels=64
        self.conv1=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
            )

        self.conv2_x=self._make_layer(block,64,num_block[0],1)
        self.conv3_x=self._make_layer(block,128,num_block[1],2)
        self.conv4_x=self._make_layer(block,256,num_block[2],2)
        self.conv5_x=self._make_layer(block,512,num_block[3],2)

        self.avg_pool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*block.expansion,num_classes)

        if init_weights:
            self._initialize_weights()

    def _make_layer(self,block,out_channels,num_blocks,stride):
        strides=[stride]+[1]*(num_blocks-1)
        layers=[]
        for stride in strides:
            layers.append(block(self.in_channels,out_channels,stride))
            self.in_channels=out_channels*block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

  # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(BottleNeck, [3,4,6,3])



#******************************  MOBILENET v1 ******************************#
class Depthwise(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6()
            )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6()
            )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Basic Conv2d
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class MobileNet(nn.Module):
    def __init__(self, width_multiplier, num_classes=50, init_weights=True):
        super().__init__()
        self.init_weights=init_weights
        alpha = width_multiplier

        self.conv1 = BasicConv2d(1, int(32*alpha), 3, stride=2, padding=1)
        self.conv2 = Depthwise(int(32*alpha), int(64*alpha), stride=1)
        # down sample
        self.conv3 = nn.Sequential(
            Depthwise(int(64*alpha), int(128*alpha), stride=2),
            Depthwise(int(128*alpha), int(128*alpha), stride=1)
            )
        # down sample
        self.conv4 = nn.Sequential(
            Depthwise(int(128*alpha), int(256*alpha), stride=2),
            Depthwise(int(256*alpha), int(256*alpha), stride=1)
            )
        # down sample
        self.conv5 = nn.Sequential(
            Depthwise(int(256*alpha), int(512*alpha), stride=2),
            Depthwise(int(512*alpha), int(512*alpha), stride=1),
            Depthwise(int(512*alpha), int(512*alpha), stride=1),
            Depthwise(int(512*alpha), int(512*alpha), stride=1),
            Depthwise(int(512*alpha), int(512*alpha), stride=1),
            Depthwise(int(512*alpha), int(512*alpha), stride=1)
            )
        # down sample
        self.conv6 = nn.Sequential(
            Depthwise(int(512*alpha), int(1024*alpha), stride=2)
            )
        # down sample
        self.conv7 = nn.Sequential(
            Depthwise(int(1024*alpha), int(1024*alpha), stride=2)
            )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(int(1024*alpha), num_classes)

        # weights initialization
        if self.init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    # weights initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def MobileNetv1(alpha=1, num_classes=25):
    return MobileNet(alpha, num_classes)

