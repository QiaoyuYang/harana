# Adapt from https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class DenseNet(nn.Module):
    def __init__(self, in_channels=12, block_depths=[3, 2, 2], growth_rates=[10, 4, 4], kernel_sizes=[7, 3, 3], bottleneck=True, drop_rate=0.6):
        super(DenseNet, self).__init__()

        self.drop_rate = drop_rate
        
        # 1st block
        in_channels_block1 = in_channels
        self.block1 = DenseBlock(block_depths[0], in_channels_block1, growth_rates[0], kernel_sizes[0], bottleneck)
        out_channels_block1 = in_channels_block1 + block_depths[0]  * growth_rates[0]

        # 2nd block
        in_channels_block2 = out_channels_block1
        self.block2 = DenseBlock(block_depths[1], in_channels_block2, growth_rates[1], kernel_sizes[1], bottleneck)
        out_channels_block2 = in_channels_block2 + block_depths[1] * growth_rates[1]


        # 3rd block
        in_channels_block3 = out_channels_block2
        self.block3 = DenseBlock(block_depths[2], in_channels_block3, growth_rates[2], kernel_sizes[2], bottleneck)
        out_channels_block3 = in_channels_block3 + block_depths[2] * growth_rates[2]

    def forward(self, x):
        
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        return out

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same")
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        
        out = self.relu(self.bn(self.conv(x)))
        
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bottleneck_scale):
        super(BottleneckBlock, self).__init__()

        inter_channels = out_channels * bottleneck_scale
        self.conv1 = nn.Conv1d(in_channels, inter_channels, kernel_size=1, padding="same") 
        self.bn1 = nn.BatchNorm1d(inter_channels)
        self.conv2 = nn.Conv1d(inter_channels, out_channels, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, block_depth, in_channels, growth_rate, kernel_size, bottleneck):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block_depth, in_channels, growth_rate, kernel_size, bottleneck)
    
    def _make_layer(self, block_depth, in_channels, growth_rate, kernel_size, bottleneck):
        
        layers = []
        for i in range(block_depth):
            if bottleneck:
                layers.append(BottleneckBlock(in_channels + i * growth_rate, growth_rate, kernel_size, 3))
            else:
                layers.append(BasicBlock(in_channels + i * growth_rate, growth_rate, kernel_size))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        return self.layer(x)
