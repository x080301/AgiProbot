import torch.nn as nn
import collections


class ResBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu2 = nn.ReLU(inplace=True)

        self.stride = stride

        self.downsample = None
        if stride != 1 or in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual

        out = self.relu2(out)

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.resblock1 = ResBlock(in_channel=64, out_channel=64, stride=1)
        self.resblock2 = ResBlock(in_channel=64, out_channel=128, stride=2)
        self.resblock3 = ResBlock(in_channel=128, out_channel=256, stride=2)
        self.resblock4 = ResBlock(in_channel=256, out_channel=512, stride=2)
        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 2)  # (512, 2)
        self.segmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        x = self.globalavgpool(x)
        x = self.flatten(x)
        # shape=x.shape
        x = self.fc(x)
        x = self.segmoid(x)

        return x
