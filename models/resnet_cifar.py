'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn.functional as F
import torch.nn.init as init
from copy import deepcopy
from torch.distributions import Distribution
from torch.nn import Module
from torch.nn.modules.container import Sequential


from layers.activation_layers import QActivation
from layers.linear_layers import _MPBaseLayer, MPConv2d, MPFC
from layers.normalization_layers import BatchNorm2d
from torch.nn import AvgPool2d
from utils.plots import plot_weight_histograms


__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def _weights_init(m):
    if isinstance(m, _MPBaseLayer):
        init.kaiming_normal_(m.weight)


class LambdaLayer(Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', use_batchnorm=True, quantize_weights = False ):
        super(BasicBlock, self).__init__()
        self.conv1 = MPConv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False, quantize_weights = quantize_weights)
        self.bn1 = BatchNorm2d(num_features=planes) if use_batchnorm else Sequential()
        self.act1 = QActivation(act_fun="relu")
        self.conv2 = MPConv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False, quantize_weights = quantize_weights )
        self.bn2 = BatchNorm2d(num_features=planes) if use_batchnorm else Sequential()
        self.shortcut = Sequential()
        self.act2 = QActivation(act_fun="relu")

        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            elif option == 'B':
                raise NotImplementedError()


    def forward(self, x):


        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            out += self.shortcut(x)

        out = self.act2(out)

        return out




    def keep_only(self, keep_element):
        # hierarchy = keep_element.split(".")
        if keep_element == "act1":
            self.conv2 = Sequential()
            self.bn2 = Sequential()
            self.shortcut = None
            self.act2 = Sequential()


def keep_only(module, keep_element):
    hierarchy = keep_element.split(".")
    for name, m in list(module.named_children())[::-1]:
        if name == hierarchy[0]:
            if len(hierarchy) > 1:
                if hasattr(m, "keep_only"):
                    m.keep_only(".".join(hierarchy[1:]))
                else:
                    keep_only(m, ".".join(hierarchy[1:]))
            break
        else:
            setattr(module, name, Sequential())


class ResNet(Module):
    def __init__(self,
                 block,
                 num_blocks,
                 quantize_weights=True,
                 quantize_activations=True,
                 use_batchnorm=True,
                 num_classes=10,
                 memory_weights_constraints_flag=True,
                 lambda_memory_weights_loss=0.1,
                 memory_weights_constraints=70.,
                 ):
        super(ResNet, self).__init__()

        self.quantize_weights = quantize_weights
        self.quantize_activations=quantize_activations

        self.name = 'RESNET20'
        self.use_batchnorm = use_batchnorm

        self.memory_weights_constraints_flag = memory_weights_constraints_flag
        if self.memory_weights_constraints_flag:
            self.lambda_memory_weights_loss = lambda_memory_weights_loss
            self.memory_weights_constraints = memory_weights_constraints


        self.in_planes = 16
        self.conv1 = MPConv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False, quantize_weights = self.quantize_weights )
        self.bn1 = BatchNorm2d(num_features=16) if use_batchnorm else Sequential()
        self.act1 = QActivation(act_fun="relu")

        self.layer1 = self._make_layer(block, 16, num_blocks[0],
                                       stride=1,use_batchnorm = self.use_batchnorm, quantize_weights = self.quantize_weights
                                       )
        self.layer2 = self._make_layer(block, 32, num_blocks[1],
                                       stride=2,use_batchnorm = self.use_batchnorm, quantize_weights = self.quantize_weights
                                       )
        self.layer3 = self._make_layer(block, 64, num_blocks[2],
                                       stride=2,use_batchnorm = self.use_batchnorm, quantize_weights = self.quantize_weights
                                       )

        self.av_pool = AvgPool2d(kernel_size=8)
        self.flatten = LambdaLayer(lambd=lambda x: x.view(x.size(0), -1))
        self.linear = MPFC(in_features=64, out_features=num_classes, quantize_weights = self.quantize_weights)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, use_batchnorm, quantize_weights ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                                use_batchnorm=use_batchnorm, quantize_weights = quantize_weights ))
            self.in_planes = planes * block.expansion

        return Sequential(*layers)

    def forward(self, x, return_dict = False):

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.av_pool(out)
        out = self.flatten(out)
        out = self.linear(out)


        if self.memory_weights_constraints_flag and self.training:
            total_memory_weights = 0
            for layer in list(self.modules()):
                if hasattr(layer, "get_memory") and layer.quantize_flag == True:
                    total_memory_weights += layer.get_memory()


            g_1 = total_memory_weights - self.memory_weights_constraints
            memory_weights_loss = self.lambda_memory_weights_loss * (torch.relu(g_1)**2 ) # lambda_1* max(0, g_1) ** 2;

            return {'out':out, 'memory_loss':memory_weights_loss, 'total_memory':total_memory_weights.detach()}


        return out

    def plot_weights(self, name_to_save):

        freq_list = []
        names_list = []
        for layer in list(self.modules()):
            if hasattr(layer, "get_memory") and layer.quantize_flag == True:
                freq_list.append(layer.plot_weights())
                # print(layer.plot_weights())
                # print(layer.name)
                names_list.append(layer.name)
        plot_weight_histograms(freq_list, names_list, name_to_save)





    def get_tb_summaries(self):
        tb_summaries = {}
        for l in [m for m in self.modules() if isinstance(m, _MPBaseLayer)]:
            tb_summaries.update(l.get_tb_summaries())
        return tb_summaries



def resnet20(*args, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3], *args, **kwargs)


def resnet32(*args, **kwargs):
    return ResNet(BasicBlock, [5, 5, 5], *args, **kwargs)


def resnet44(*args, **kwargs):
    return ResNet(BasicBlock, [7, 7, 7], *args, **kwargs)


def resnet56(*args, **kwargs):
    return ResNet(BasicBlock, [9, 9, 9], *args, **kwargs)


def resnet110(*args, **kwargs):
    return ResNet(BasicBlock, [18, 18, 18], *args, **kwargs)


def resnet1202(*args, **kwargs):
    return ResNet(BasicBlock, [200, 200, 200], *args, **kwargs)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":
    net = resnet20()
    out = net(torch.randn(11, 3, 32, 32))[0]
    print(out)
