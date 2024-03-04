

from __future__ import print_function, division, absolute_import
from collections import OrderedDict
from ..builder import BACKBONES
from mmcv.runner import load_checkpoint
import logging
from mmcv.cnn import constant_init, kaiming_init
import math
from torchdiffeq import odeint, odeint_adjoint
 
import torch.nn as nn
from torch.utils import model_zoo

import torch
import torch.nn.functional as F
__all__ = ['SENet', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152',
           'se_resnext50_32x4d', 'se_resnext101_32x4d']
 
pretrained_settings = {
            
    'senet154': {
            
        'imagenet': {
            
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet50': {
            
        'imagenet': {
            
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet101': {
            
        'imagenet': {
            
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet152': {
            
        'imagenet': {
            
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext50_32x4d': {
            
        'imagenet': {
            
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext101_32x4d': {
            
        'imagenet': {
            
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}
 
 
# class SEModule(nn.Module):
 
#     def __init__(self, channels, reduction):
#         super(SEModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
#                              padding=0)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
#                              padding=0)
#         self.sigmoid = nn.Sigmoid()
 
#     def forward(self, x):
#         module_input = x
#         x = self.avg_pool(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)
#         return module_input * x
MAX_NUM_STEPS = 1000
class ODEBlock(nn.Module):
    def __init__(self, odefunc, tol=1e-3, adjoint=False):
        """
        Code adapted from https://github.com/EmilienDupont/augmented-neural-odes

        Utility class that wraps odeint and odeint_adjoint.

        Args:
            odefunc (nn.Module): the module to be evaluated
            tol (float): tolerance for the ODE solver
            adjoint (bool): whether to use the adjoint method for gradient calculation
        """
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, eval_times=None):
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.nfe = 0

        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float()
        else:
            integration_time = eval_times.type_as(x)
            
        if self.adjoint:
            out = odeint_adjoint(self.odefunc, x, integration_time,
                                 rtol=self.tol, atol=self.tol, method='dopri5',
                                 options={'max_num_steps': MAX_NUM_STEPS})
        else:
            out = odeint(self.odefunc, x, integration_time,
                         rtol=self.tol, atol=self.tol, method='dopri5',
                         options={'max_num_steps': MAX_NUM_STEPS})

        if eval_times is None:
            return out[1]  # Return only final time
        else:
            return out[1]

    def trajectory(self, x, timesteps):
        integration_time = torch.linspace(0., 1., timesteps)
        return self.forward(x, eval_times=integration_time)
    
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, non_linearity='relu'):
        """
        Block for ConvODEUNet

        Args:
            in_channels (int): number of filters for the conv layers
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
        """
        super(ConvBlock, self).__init__()

        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(in_channels) #归一化/正则化
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.non_linearity = get_nonlinearity(non_linearity) #激活函数选择模块

    def forward(self, x):
        out = self.norm1(x)
        out = self.conv1(out)
        out = self.non_linearity(out)
        out = self.norm2(out)
        out = self.conv2(out)
        out = self.non_linearity(out)
        return out
    
def get_nonlinearity(name):
    """Helper function to get non linearity module, choose from relu/softplus/swish/lrelu"""
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'softplus':
        return nn.Softplus()
    elif name == 'swish':
        return Swish(inplace=True)
    elif name == 'lrelu':
        return nn.LeakyReLU()

class nmODE2(nn.Module):
    def __init__(self):
        """
        """
        super(nmODE2, self).__init__()
        self.nfe = 0  # Number of function evaluations
        self.gamma = None
    
    def fresh(self, gamma):
        self.gamma = gamma
    
    def forward(self, t, p):
        self.nfe += 1
        dpdt = -p + torch.pow(torch.sin(p + self.gamma), 2) #计算指数
        # dpdt = -p + torch.pow(torch.sin(p + torch.cos(p + self.gamma)), 2)
        return dpdt
 
 
 
class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out = out + residual
        out = self.relu(out)
 
        return out
 
 
class ODEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4
 
    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(ODEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        # self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride
 
 
class ODEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4
 
    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(ODEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        # +ode
        # self.eval_times = torch.tensor(eval_times).float().cuda()
        # # self.input_1x1 = nn.Conv2d(in_channels, nf, 1, 1)
        # self.conv1_ode = ConvBlock(planes, non_linearity, stride) #
        # self.nmODE_down1 = nmODE2()
        # self.ode_down1 = ODEBlock(self.nmODE_down1, tol=tol, adjoint=adjoint)
        # self.non_linearity = get_nonlinearity(non_linearity)

        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        # +ode
        # self.conv2_ode = ConvBlock(planes, non_linearity)
        # self.nmODE_down2 = nmODE2()
        # self.ode_down2 = ODEBlock(self.nmODE_down2, tol=tol, adjoint=adjoint)


        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # +ode
        # self.conv2_ode = ConvBlock(planes, non_linearity)
        # self.nmODE_down2 = nmODE2()
        # self.ode_down2 = ODEBlock(self.nmODE_down2, tol=tol, adjoint=adjoint)


        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        # self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride
 
 
class ODEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4
 
    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(ODEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        # self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride
 
 
bottleneck_dic = {
            
    'ODEBottleneck': ODEBottleneck,
    'ODEResNetBottleneck': ODEResNetBottleneck,
    'ODEResNeXtBottleneck': ODEResNeXtBottleneck
}
 
# 注册当前模块
@BACKBONES.register_module 
class ODENet03024(nn.Module):
 
    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1,norm_eval=True, frozen_stages=-1, zero_init_residual=True, num_classes=1000, eval_times = (0, 1),
                 non_linearity='softplus',
                 tol=1e-3,
                 adjoint=False,
                 ):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(ODENet03024, self).__init__()
        block = bottleneck_dic[block]
        self.inplanes = inplanes
        self.inplanes1 = inplanes / 2
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        # +ode
        self.eval_times = torch.tensor(eval_times).float().cuda()
        # self.input_1x1 = nn.Conv2d(in_channels, nf, 1, 1)
        self.conv1_ode = ConvBlock(int(self.inplanes), non_linearity) #
        self.nmODE_down1 = nmODE2()
        self.ode_down1 = ODEBlock(self.nmODE_down1, tol=tol, adjoint=adjoint)
        self.non_linearity = get_nonlinearity(non_linearity)

        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        # +ode
        self.conv2_ode = ConvBlock(int(self.inplanes), non_linearity)
        self.nmODE_down2 = nmODE2()
        self.ode_down2 = ODEBlock(self.nmODE_down2, tol=tol, adjoint=adjoint)



        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        # +ode
        self.conv3_ode = ConvBlock(int(self.inplanes), non_linearity )
        self.nmODE_down3 = nmODE2()
        self.ode_down3 = ODEBlock(self.nmODE_down3, tol=tol, adjoint=adjoint)




        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2, 
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        # +ode
        self.conv4_ode = ConvBlock(int(self.inplanes), non_linearity)
        self.nmODE_down4 = nmODE2()
        self.ode_down4 = ODEBlock(self.nmODE_down4, tol=tol, adjoint=adjoint)

        # self.avg_pool = nn.AvgPool2d(7, stride=1)
        # self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        # self.last_linear = nn.Linear(512 * block.expansion, num_classes)
        self._freeze_stages()
 
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for m in [self.layer0]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
 
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                if self.zero_init_residual:
                    for m in self.modules():
                        if isinstance(m, Bottleneck):
                            constant_init(m.bn3, 0)
        else:
            raise TypeError('pretrained must be a str or None')
 
    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
 
        return nn.Sequential(*layers)
 
    def features(self, x):
        outputs = []
        x = self.layer0(x)
        x = self.layer1(x)
        #+ode
#         x = self.non_linearity(x)
# #         features1 = self.conv1_ode(x)
#         features1 = x
#         self.nmODE_down1.fresh(features1)
#         features1 = self.ode_down1(torch.zeros_like(features1), self.eval_times)
#         # x = self.non_linearity(self.conv1_2(features1))
#         x = self.non_linearity(features1)
        # x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        outputs.append(x)
        x = self.layer2(x)
        # +ode
# #         features2 = self.conv2_ode(x)
#         features2 = x
#         self.nmODE_down2.fresh(features2)
#         features2 = self.ode_down2(torch.zeros_like(features2), self.eval_times)
#         x = self.non_linearity(features2)
        # x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        outputs.append(x)
        x = self.layer3(x)
        # +ode
# #         features3 = self.conv3_ode(x)
#         features3 = x
#         self.nmODE_down3.fresh(features3)
#         features3 = self.ode_down3(torch.zeros_like(features3), self.eval_times)
#         x = self.non_linearity(features3)
        # x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        outputs.append(x)
        x = self.layer4(x)
        # +ode
#         features4 = self.conv4_ode(x)
        features4 = x
        self.nmODE_down4.fresh(features4)
        features4 = self.ode_down4(torch.zeros_like(features4), self.eval_times)
        x = self.non_linearity(features4)
        # x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        outputs.append(x)
        return x, outputs
    '''    
    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x
    '''
    def forward(self, x):
        x, outputs = self.features(x)
        # x = self.logits(x)
        return outputs  # x
 
    def train(self, mode=True):
        super(ODENet03024, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, (nn.BatchNorm2d)):
                    m.eval()
 
def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
 
 
def ODEnet154(num_classes=1000, pretrained='imagenet'):
    model = ODENet03024(ODEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
                  dropout_p=0.2, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['senet154'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model
 
 
def ODE_resnet50(num_classes=1000, pretrained='imagenet'):
    model = ODENet03024(ODEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet50'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model
 
 
def ODE_resnet101(num_classes=1000, pretrained='none'):
    model = ODENet03024(ODEResNetBottleneck, [3, 4, 23, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet101'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model
 
 
def ODE_resnet152(num_classes=1000, pretrained='imagenet'):
    model = ODENet03024(ODEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet152'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model
 

def ODE_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = ODENet03024(ODEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model
 

def ODE_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = ODENet03024(ODEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model