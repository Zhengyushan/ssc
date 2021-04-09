"""
Date: 2021/04
Author:Yushan Zheng
Email:yszheng@buaa.edu.cn
"""

import os
import torch
import torch.nn as nn
from collections import OrderedDict

EPSILON = 0.00001
PIXEL_MIN_VALUE = 1.0/255.0

class StainStdCapsule(nn.Module):
    def __init__(self, routing_iter=3, stain_type=2, group_num=5, group_width=3):
        super(StainStdCapsule, self).__init__()
        self.group_num = group_num
        self.stain_type = stain_type
        self.routing_iter = routing_iter
        self.width = group_width
        self.debug_counter = 0


        self.stain_presep = nn.Sequential(OrderedDict([
                ('ssc_conv0', nn.Conv2d(3, group_num*group_width
                    , kernel_size=1, bias=True, padding=0)),
                ('ssc_act0', nn.LeakyReLU()),
            ]))
        self.projection = nn.Sequential(OrderedDict([
                ('ssc_conv1',nn.Conv2d(group_num*group_width, group_num*stain_type, 
                    kernel_size=1, bias=False, padding=0, groups=group_num)),
                ('ssc_act1', nn.LeakyReLU()),
            ]))
        self.reconstruction = nn.Sequential(OrderedDict([
                ('ssc_conv_re', nn.Conv2d(stain_type, 3
                    , kernel_size=1, bias=True, padding=0)),
                ('ssc_bn_re', nn.BatchNorm2d(3)),
                ('ssc_act_re', nn.LeakyReLU()),
            ]))


    def forward(self, input_tensor):
        od_input = -torch.log((input_tensor + PIXEL_MIN_VALUE))
        x = self.stain_presep(od_input)
        x = self.projection(x)
        x = x.reshape(x.size(0), self.group_num, self.stain_type, x.size(2), x.size(3))
        c = self.sparsity_routing(x)

        output = torch.sum(x * c, dim=1)
        re_image = self.reconstruction(output)
        re_image = torch.exp(-re_image)

        return output, re_image


    def sparsity_routing(self, input_tensor):
        u = input_tensor.data
        s = u
        b = 0.0
        for _ in range(self.routing_iter-1):
            b = b + self.pixel_sparsity(s) + self.channel_sparsity(s)
            c = b.softmax(dim=1)
            s = torch.sum(c * u, dim=1, keepdim=True)
            s = s + u

        score = self.pixel_sparsity(s) + self.channel_sparsity(s)
        b = b + score
        c = b.softmax(dim=1)

        return c


    def pixel_sparsity(self, group_stains):
        values = group_stains + EPSILON
        l2 = values.pow(2).sum(dim=2, keepdim=True)
        l2 = l2.sqrt() + EPSILON
        l1 = values.abs().sum(dim=2, keepdim=True)
        sqrt_n = self.stain_type ** 0.5
        sparsity = (sqrt_n - l1 / l2) / (sqrt_n - 1)
        sparsity = sparsity.mean(dim=(3,4), keepdim=True)

        return sparsity


    def channel_sparsity(self, group_stains):
        values = group_stains + EPSILON
        l2 = values.pow(2).sum(dim=(3,4), keepdim=True)
        l2 = l2.sqrt() + EPSILON
        l1 = values.abs().sum(dim=(3,4), keepdim=True)
        sqrt_n = (group_stains.size(3) * group_stains.size(4)) ** 0.5
        sparsity = (sqrt_n - l1 / l2) / (sqrt_n - 1)
        sparsity = sparsity.mean(dim=2, keepdim=True)

        return sparsity
