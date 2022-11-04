# Adapt from https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class DenseNet(nn.Module):
    def __init__(self, num_pc, block_depths=[3, 2, 2], growth_rates=[10, 4, 4], kernel_sizes=[7, 3, 3], bottleneck=True, drop_rate=0.0):
        super(DenseNet, self).__init__()

        if bottleneck == True:
            block = BottleneckBlock
        else:
            block = BasicBlock

        block = BasicBlock
        
        self.num_pc = num_pc
        # 1st conv before any dense block
        self.input_trans = PCwiseConv1d(num_pc, in_planes=1, out_planes=growth_rates[0], kernel_size=3, drop_rate=drop_rate)
        
        # 1st block
        self.block1 = DenseBlock(num_pc, block_depths[0], growth_rates[0], growth_rates[0], kernel_sizes[0], block, drop_rate)
        self.trans1 = TransitionBlock(num_pc, (block_depths[0] + 1) * growth_rates[0], growth_rates[1], kernel_size=3, drop_rate=drop_rate)

        # 2nd block
        self.block2 = DenseBlock(num_pc, block_depths[1], growth_rates[1], growth_rates[1], kernel_sizes[1], block, drop_rate)
        self.trans2 = TransitionBlock(num_pc, (block_depths[1] + 1) * growth_rates[1], growth_rates[2], kernel_size=3, drop_rate=drop_rate)


        # 3rd block
        self.block3 = DenseBlock(num_pc, block_depths[2], growth_rates[2], growth_rates[2], kernel_sizes[2], block, drop_rate)
        self.output_bn = nn.BatchNorm1d((block_depths[2] + 1) * growth_rates[2])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        out = self.input_trans(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = torch.stack([self.output_bn(out[:, pc, :, :]) for pc in range(self.num_pc)], dim=1)

        return self.relu(out)

class PCwiseConv1d(nn.Module):
    def __init__(self, num_pc, in_planes, out_planes, kernel_size, drop_rate=0.0):
        super(PCwiseConv1d, self).__init__()
        self.drop_rate = drop_rate

        self.conv_all_pc = []
        self.conv_all_pc = nn.ModuleList([nn.Conv1d(in_planes, out_planes, kernel_size, stride=1, padding="same", bias=False) for i in range(num_pc)])

    def forward(self, x):
        out = torch.stack([conv_one_pc(x[:, pc, :, :]) for pc, conv_one_pc in enumerate(self.conv_all_pc)], dim=1)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out

class BasicBlock(nn.Module):
    def __init__(self, num_pc, in_planes, out_planes, kernel_size, drop_rate=0.0):
        super(BasicBlock, self).__init__()

        self.num_pc = num_pc
        self.drop_rate = drop_rate

        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = PCwiseConv1d(num_pc, in_planes=in_planes, out_planes=out_planes, kernel_size=kernel_size, drop_rate=drop_rate)
    
    def forward(self, x):

        out = torch.stack([self.bn1(x[:, pc, :, :]) for pc in range(self.num_pc)], dim=1)
        out = self.conv1(self.relu(out))

        return torch.cat([x, out], 2)

class BottleneckBlock(nn.Module):
    def __init__(self, num_pc, in_planes, out_planes, kernel_size, bottleneck_scale, drop_rate=0.0):
        super(BottleneckBlock, self).__init__()
        
        inter_planes = out_planes * bottleneck_scale
        
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = PCwiseConv1d(num_pc, in_planes, inter_planes, kernel_size=1, drop_rate=drop_rate) 
        self.bn2 = nn.BatchNorm1d(inter_planes)
        self.conv2 = PCwiseConv1d(num_pc, inter_planes, out_planes, kernel_size=3, drop_rate=drop_rate)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        
        out = torch.stack([self.bn1(x[:, pc, :, :]) for pc in range(self.num_pc)], dim=1)
        out = self.conv1(self.relu(out))
        out = torch.stack([self.bn2(out[:, pc, :, :]) for pc in range(self.num_pc)], dim=1)
        out = self.conv2(self.relu(out))
        
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, num_pc, in_planes, out_planes, kernel_size, drop_rate=0.0):
        super(TransitionBlock, self).__init__()

        self.num_pc = num_pc
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = PCwiseConv1d(num_pc, in_planes=in_planes, out_planes=out_planes, kernel_size=kernel_size, drop_rate=drop_rate)
    
    def forward(self, x):
        out = torch.stack([self.bn1(x[:, pc, :, :]) for pc in range(self.num_pc)], dim=1)
        out = self.conv1(self.relu(x))

        return out

class DenseBlock(nn.Module):
    def __init__(self, num_pc, block_depth, in_planes, growth_rate, kernel_size, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(num_pc, block, block_depth, in_planes, growth_rate, kernel_size, dropRate)
    
    def _make_layer(self, num_pc, block, block_depth, in_planes, growth_rate, kernel_size, dropRate):
        layers = []
        for i in range(block_depth):
            layers.append(block(num_pc, in_planes + i * growth_rate, growth_rate, kernel_size, dropRate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)
