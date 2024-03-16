import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init, Sequential
import math

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

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

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha

        # Original weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # Low-rank matrices
        self.delta_W = nn.Parameter(torch.Tensor(rank, in_features))
        self.delta_B = nn.Parameter(torch.Tensor(out_features, rank))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.delta_W, std=0.01)
        nn.init.normal_(self.delta_B, std=0.01)

    def forward(self, x):
        W = self.weight + self.alpha * (self.delta_B @ self.delta_W)
        return F.linear(x, W, self.bias)

class LoRAConv(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride=1, padding=0, groups=1, bias=False, lora_rank=4, lora_alpha=2.0):
        super(LoRAConv, self).__init__()
        # 正常的卷積操作
        self.conv = nn.Conv2d(c1, c2, kernel_size, stride, padding, groups=groups, bias=bias)
        # LoRA層
        self.lora = LoRALayer(c2, c2, rank=lora_rank, alpha=lora_alpha)

    def forward(self, x):
        x = self.conv(x)
        x = self.lora(x)
        return x
    
class C3lora(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, lora_rank=4, lora_alpha=2.0):
        super(C3lora, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = LoRAConv(c1, c_, 1, 1, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.cv2 = LoRAConv(c1, c_, 1, 1, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.cv3 = LoRAConv(2 * c_, c2, 1, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # 注意：您也可以在Bottleneck或CrossConv中集成LoRA，取決於您的具體需求

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
    

class LoRAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, rank=4, alpha=1.0):
        super(LoRAConv2d, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # 標準卷積層
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)

        # LoRA低秩矩陣
        self.delta_W = nn.Parameter(torch.Tensor(rank, out_channels, in_channels // groups, kernel_size, kernel_size))
        self.delta_B = nn.Parameter(torch.Tensor(out_channels, rank))
        
        # 如果需要偏置
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.normal_(self.delta_W, std=0.01)
        nn.init.normal_(self.delta_B, std=0.01)

    def forward(self, x):
        # 原始卷積操作
        out_conv = self.conv(x)
        
        # 计算低秩更新
        delta_W = self.delta_B.view(self.conv.out_channels, self.rank, 1, 1) * self.delta_W.view(1, self.rank, *self.delta_W.shape[1:])
        delta_W = delta_W.sum(dim=1)  # 將秩維度合併
        
        # 應用低秩更新
        out_lora = F.conv2d(x, delta_W, None, self.stride, self.padding, self.dilation, self.groups)
        
        # 結合原始卷積和低秩更新的結果
        out = out_conv + self.alpha * out_lora
        
        if self.bias is not None:
            out += self.bias.view(1, self.bias.size(0), 1, 1)
        
        return out
