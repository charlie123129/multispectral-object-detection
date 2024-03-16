import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        # print(c1, c2, k, s,)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # print("Conv", x.shape)
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

def autopad(k, p=None):
    # Calculate padding automatically, same as in the Conv class
    if p is None:
        p = k // 2
    return p

class MiniUNet(nn.Module):
    """简化版的U-Net作为卷积层的替代"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1):
        super(MiniUNet, self).__init__()
        padding = autopad(kernel_size, padding)
        self.down_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.down_conv2 = nn.Conv2d(out_channels, out_channels*2, kernel_size, stride, padding, groups=groups, bias=False)
        self.up_conv1 = nn.ConvTranspose2d(out_channels*2, out_channels, kernel_size=2, stride=2)
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # Assuming activation is always SiLU for simplicity

    def forward(self, x):
        x1 = self.act(self.bn(self.down_conv1(x)))
        x2 = self.act(self.bn(self.down_conv2(x1)))
        x2 = F.interpolate(x2, scale_factor=2, mode='nearest')
        x2 = self.act(self.bn(self.up_conv1(x2)))
        x = x1 + x2  # 跳跃连接
        x = self.act(self.bn(self.out_conv(x)))
        return x

class ConvUNet(nn.Module):
    """使用MiniUNet替代Conv的U-Net结构"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(ConvUNet, self).__init__()
        self.unet = MiniUNet(c1, c2, k, s, p, g)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.unet(x)
        return self.act(x)



'''
class MiniUNet11(nn.Module):
    #"""简化版的U-Net作为卷积层的替代"""
    def __init__(self, in_channels, out_channels):
        super(MiniUNet, self).__init__()
        self.down_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.down_conv2 = nn.Conv2d(out_channels, out_channels*2, kernel_size=3, padding=1)
        self.up_conv1 = nn.ConvTranspose2d(out_channels*2, out_channels, kernel_size=2, stride=2)
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x1 = F.relu(self.down_conv1(x))
        x2 = F.relu(self.down_conv2(x1))
        x2 = F.interpolate(x2, scale_factor=2, mode='nearest')
        x2 = self.up_conv1(x2)
        x = x1 + x2  # 跳跃连接
        x = self.out_conv(x)
        return x

class ConvUNet11(nn.Module):
    #"""在U-Net结构中使用MiniUNet替代标准的卷积层"""
    def __init__(self, n_channels, n_classes):
        super(ConvUNet, self).__init__()
        # 使用MiniUNet替代标准的卷积层
        self.inc = MiniUNet(n_channels, 64)
        self.down1 = MiniUNet(64, 128)
        self.down2 = MiniUNet(128, 256)
        self.down3 = MiniUNet(256, 512)
        self.down4 = MiniUNet(512, 1024)
        self.up1 = MiniUNet(1024, 512)
        self.up2 = MiniUNet(512, 256)
        self.up3 = MiniUNet(256, 128)
        self.up4 = MiniUNet(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
'''