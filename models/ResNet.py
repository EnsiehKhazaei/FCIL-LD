import torch.nn as nn
import torch.nn.functional as F
import math


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, init_weight=True, cifar=True):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if cifar:
            self.feature = nn.AvgPool2d(4, stride=1)
        else:
            self.feature = nn.AvgPool2d(8)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.feature(out)
        out = out.view(out.size(0), -1)
        return out


def ResNet18(num_classes, cifar):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, cifar=cifar)


def ResNet34(num_classes, cifar=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, cifar=cifar)


import torch
import torch.nn as nn
import torchvision.models as models
class ResNets(nn.Module):
    def __init__(self, args, use_pretrained=True):
        super(ResNets, self).__init__()
        # print(args.model)
        if use_pretrained is True:
            if args.model == "resnet50":
                resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            if args.model == "resnet34":
                resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            if args.model == "resnet18":
                resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            if args.model == "mobilenet":
                resnet = models.mobilenet_v2(pretrained=True)
            
        else:
            if args.model == "resnet50":
                resnet = models.resnet50(weights=None)
            if args.model == "resnet34":
                resnet = models.resnet34(weights=None)
            if args.model == "resnet18":
                resnet = models.resnet18(weights=None)
            if args.model == "mobilenet":
                resnet = models.mobilenet_v2(weights=None)
            
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        self.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=10)
     
    def forward(self, x):
        x = self.features(x)
        x = x.squeeze()
        return x