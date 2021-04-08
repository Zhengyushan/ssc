"""
Date: 2021/04
Author:Yushan Zheng
Email:yszheng@buaa.edu.cn
"""
import torch
import torchvision.models as models
from ssc_module import StainStdCapsule


class ApplicationWithSSC(torch.nn.Module):
    def __init__(self, cnn_arch_name, num_classes, args=None):
        super(ApplicationWithSSC, self).__init__()
        
        print("=> creating ssc module")
        self.ssc_module = StainStdCapsule(
                routing_iter=args.num_routings,
                stain_type=args.num_stains,
                group_num=args.num_groups, 
                group_width=args.group_width,
            )

        print("=> creating cnn module '{}'".format(cnn_arch_name))
        self.app_module = models.__dict__[cnn_arch_name](num_classes=num_classes)
        
        # Here, we need to change the input channels from 3 to the number of stains
        # by redefining the first convolution layer of the CNN backbone.
        # Different CNN backbones have different definitions of the first layer.
        # The code below is specific for DenseNet series.
        self.app_module.features.conv0 = torch.nn.Conv2d(args.num_stains, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)

    def forward(self, input_tensor):
        normed_input, reconst = self.ssc_module(input_tensor)
        output = self.app_module(normed_input)

        return output, reconst