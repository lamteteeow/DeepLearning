import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


## Implementing the ResNet block ##


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )

        self.bn1 = nn.BatchNorm2d(num_features=in_channels)

        self.conv2 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=0
        )

        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.identity_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride
        )
        self.identity_bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        skip_inp = x.clone()

        x = self.conv1(x)
        # x = self.dropout(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout(x)

        skip_inp = self.identity_conv(skip_inp)

        if skip_inp.shape != x.shape:
            skip_inp = F.interpolate(skip_inp, size=x.shape[2:])

        skip_inp = self.identity_bn(skip_inp)

        x = x + skip_inp

        x = self.relu(x)

        return x


## Creating the ResNet ##


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet_head = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            ResBlock(64, 64, 1),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            ResBlock(256, 512, 2),
        )

        # [info](https://discuss.pytorch.org/t/global-average-pooling-in-pytorch/6721/10) 
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.avg_pool = nn.AvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet_head(x)
        x = self.dropout(x)
        x = self.avg_pool(x)
        x = self.flat(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x
