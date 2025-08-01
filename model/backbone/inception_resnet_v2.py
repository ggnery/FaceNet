import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t

class InceptionResNetV2(nn.Module):
    def __init__(self, device: torch.device, embedding_size=512, dropout_keep=0.8) -> None:
        super(InceptionResNetV2, self).__init__()
        self.embedding_size = embedding_size
        self.dropout_prob = 1 - dropout_keep
        self.device = device
        
        #Stem
        self.stem = Stem(device)
        
        #5 x InceptionResNetA
        self.resnet_a_blocks = nn.Sequential(
            *[InceptionResNetA(384, 0.17, device) for _ in range(5)]
        )     
        self.reduction_a = ReductionA(384, k = 256, l = 256, m = 384, n = 384, device=device)
            
        # 10 x InceptionResNetB
        self.resnet_b_blocks = nn.Sequential(
            *[InceptionResNetB(1152,  0.1, device) for _ in range (10)]
        )
        
        self.reduction_b = ReductionB(1152, device)
        
        # 5 x InceptionResNet5
        self.resnet_c_blocks = nn.Sequential(
            *[InceptionResNetC(2144,  0.2, device) for _ in range (5)]
        )
        
        # AvgPool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout 
        self.dropout = nn.Dropout(self.dropout_prob)
        
        # Linear
        self.linear = nn.Sequential(
            nn.Linear(2144, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)
        resnet_a = self.resnet_a_blocks(stem) 
        reduction_a = self.reduction_a(resnet_a)
        resnet_b = self.resnet_b_blocks(reduction_a)
        reduction_b = self.reduction_b(resnet_b) 
        resnet_c = self.resnet_c_blocks(reduction_b)
        
        avgpool = self.avgpool(resnet_c)
        avgpool = torch.flatten(avgpool, 1)

        dropout = self.dropout(avgpool)
        linear = self.linear(dropout)
        return linear

class Stem(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super(Stem, self).__init__()
        
        # Input Shape is 299 x 299 x 3
        self.seq1 = nn.Sequential(
            Conv2dBatchNormalized(3, 32, kernel_size=3, stride=2, padding="valid", device=device), # Output: 149x149x32
            Conv2dBatchNormalized(32, 32, kernel_size=3, padding="valid", device=device), # Output: 147x147x32
            Conv2dBatchNormalized(32, 64, kernel_size=3, device=device) # Output: 147x147x64 
        )
        
        # (Branch1) output: 73x73x160
        self.branch1_pool = nn.MaxPool2d(3, stride=2, padding=0) 
        self.branch1_conv = Conv2dBatchNormalized(64, 96, kernel_size=3, stride=2, padding="valid",  device=device) 
        
        # (Branch2) output: 71x71x192
        self.branch2_seq1 =nn.Sequential(
            Conv2dBatchNormalized(160, 64, kernel_size=1, device=device), 
            Conv2dBatchNormalized(64, 96, kernel_size=3, padding="valid", device=device) 
        )
        self.branch2_seq2 =nn.Sequential(
            Conv2dBatchNormalized(160, 64, kernel_size=1, device=device),
            Conv2dBatchNormalized(64, 64, kernel_size=(7,1), device=device),
            Conv2dBatchNormalized(64, 64, kernel_size=(1,7), device=device),
            Conv2dBatchNormalized(64, 96, kernel_size=3, padding="valid", device=device) 
        )
        
        # (Branch3) output: 35x35x384
        self.branch3_conv = Conv2dBatchNormalized(192, 192, kernel_size=3, stride=2, padding="valid", device=device) # THE ARTICLE DIDNT EXPLICITLY SAID THIS HAS STRIDE=2!!!
        self.branch3_pool = nn.MaxPool2d(3, stride=2, padding=0) 
    
    def forward(self, x: torch.tensor):
        x = self.seq1(x) # Output: 147x147x64
        x = torch.cat([self.branch1_pool(x), self.branch1_conv(x)], dim=1) # Output: 73x73x160
        x = torch.cat([self.branch2_seq1(x), self.branch2_seq2(x)], dim=1) # Output: 71x71x192
        x = torch.cat([self.branch3_conv(x), self.branch3_pool(x)], dim=1) # Output: 35x35x384
        return x 

class InceptionResNetA(nn.Module):
    def __init__(self, in_channels: int, scale: float = 0.17, device: torch.device = None) -> None:
        super(InceptionResNetA, self).__init__()     
        self.scale = scale
        self.branch1= Conv2dBatchNormalized(in_channels, 32, 1, device=device) 
        
        self.branch2= nn.Sequential(
            Conv2dBatchNormalized(in_channels, 32, kernel_size=1, device=device), 
            Conv2dBatchNormalized(32, 32, kernel_size=3, device=device)
        )   
        
        self.branch3= nn.Sequential(
            Conv2dBatchNormalized(in_channels, 32, kernel_size=1, device=device),
            Conv2dBatchNormalized(32, 48, kernel_size=3, device=device),
            Conv2dBatchNormalized(48, 64, kernel_size=3, device=device)
        ) 
        
        self.linear_conv = nn.Conv2d(128, in_channels, kernel_size=1, padding="same", bias=True, device=device)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.branch1(x) # Output: 35x35x32
        branch2 = self.branch2(x) # Output: 35x35x32
        branch3 = self.branch3(x) # Output: 35x35x64
        
        mixed = torch.cat([branch1, branch2, branch3], dim=1) # Output: 35x35x96
        linear_out = self.linear_conv(mixed) # Output: 35x35x384
        
        residual_sum = linear_out * self.scale + x
        
        return self.relu(residual_sum)
        
class ReductionA(nn.Module):
    def __init__(self, in_channels: int, k: int = 256, l: int = 256, m: int = 384, n: int = 384, device: torch.device = None) -> None:
        super(ReductionA, self).__init__()
        self.branch1 = nn.MaxPool2d(3, stride=2, padding=0)
        self.branch2 = Conv2dBatchNormalized(in_channels, n, kernel_size=3, padding="valid", stride = 2, device = device)
        self.branch3 = nn.Sequential(
            Conv2dBatchNormalized(in_channels, k, kernel_size=1, device = device),
            Conv2dBatchNormalized(k, l, kernel_size=3, device = device),
            Conv2dBatchNormalized(l, m, kernel_size=3, padding="valid", stride=2, device = device)
        ) 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        
        return torch.cat([branch1, branch2, branch3], dim=1)
    
    
class InceptionResNetB(nn.Module):
    def __init__(self, in_channels: int, scale: float = 0.10, device: torch.device = None) -> None:
        super(InceptionResNetB, self).__init__()
        self.scale = scale
        self.branch1 =  Conv2dBatchNormalized(in_channels, 192, kernel_size=1, device=device)
        self.branch2 = nn.Sequential(
            Conv2dBatchNormalized(in_channels, 128, kernel_size=1, device=device),
            Conv2dBatchNormalized(128, 160, kernel_size=(1,7), device=device),
            Conv2dBatchNormalized(160, 192, kernel_size=(7,1), device=device)
        )
        
        self.linear_conv = nn.Conv2d(384, in_channels, kernel_size=1, padding="same", bias=True, device=device)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.branch1(x) 
        branch2 = self.branch2(x) 
        
        mixed = torch.cat([branch1, branch2], dim=1)
        linear_out = self.linear_conv(mixed)
        
        residual_sum = linear_out * self.scale + x
        return self.relu(residual_sum)
    
class ReductionB(nn.Module):
    def __init__(self, in_channels: int, device: torch.device = None) -> None:
        super(ReductionB, self).__init__()
        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.branch2 = nn.Sequential(
            Conv2dBatchNormalized(in_channels, 256, kernel_size=1, device=device),
            Conv2dBatchNormalized(256, 384, kernel_size=3, padding="valid", stride=2, device=device)
        )
        self.branch3 = nn.Sequential(
            Conv2dBatchNormalized(in_channels, 256, kernel_size=1, device=device),
            Conv2dBatchNormalized(256, 288, kernel_size=3, padding="valid", stride=2, device=device)
        )
        self.branch4 = nn.Sequential(
            Conv2dBatchNormalized(in_channels, 256, kernel_size=1, device=device),
            Conv2dBatchNormalized(256, 288, kernel_size=3, device=device),
            Conv2dBatchNormalized(288, 320, kernel_size=3, padding="valid", stride=2, device=device)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.cat([branch1, branch2, branch3, branch4], dim=1)
    
class InceptionResNetC(nn.Module):
    def __init__(self, in_channels: int, scale: float = 0.2, device: torch.device = None) -> None:
        super(InceptionResNetC, self).__init__()
        self.scale = scale
        self.branch1 = Conv2dBatchNormalized(in_channels, 192, kernel_size=1, device=device)
        self.branch2 = nn.Sequential(
            Conv2dBatchNormalized(in_channels, 192, kernel_size=1, device=device),
            Conv2dBatchNormalized(192, 224, kernel_size=(1,3), device=device),
            Conv2dBatchNormalized(224, 256, kernel_size=(3,1), device=device),
        )
        
        self.linear_conv = nn.Conv2d(448, in_channels, kernel_size=1, padding="same", bias=True, device=device)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        
        mixed = torch.cat([branch1, branch2], dim=1)
        linear_out = self.linear_conv(mixed)
        
        residual_sum = linear_out * self.scale + x
        return self.relu(residual_sum)
        
class Conv2dBatchNormalized(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: _size_2_t, 
                 padding: _size_2_t | str = "same", 
                 stride: _size_2_t = 1,
                 device: torch.device = None,
                 has_bias: bool = False) -> None:

        super(Conv2dBatchNormalized, self).__init__() 
        self.conv2d_bn = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=has_bias, device=device),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
        ) 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d_bn(x)
    
    