import torch
from torch import nn
from torch.nn import functional as F


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


# 下采样
class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


# 上采样
class UPSample(nn.Module):
    def __init__(self, channel):
        super(UPSample, self).__init__()
        self.layer = nn.ConvTranspose2d(channel, channel // 2, 2, stride=2, padding=0, output_padding=0, bias=False)

    def forward(self, x, feature_map):
        up = self.layer(x)
        return torch.cat((up, feature_map), dim=1)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 编码器
        self.c1 = Conv_Block(1, 64)  # 输入通道数为 1
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)
        # 解码器
        self.u1 = UPSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.u2 = UPSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UPSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UPSample(128)
        self.c9 = Conv_Block(128, 64)
        # 输出层
        self.out = nn.Conv2d(64, 1, 3, 1, 1)  # 输出通道数为 1
        self.TH = nn.Sigmoid()

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.TH(self.out(O4))


if __name__ == '__main__':
    x = torch.randn(2, 1, 128, 128)  # 输入通道数为 1
    print(x.dtype)
    net = UNet()
    print(net(x).shape)
    print(net(x).dtype)
