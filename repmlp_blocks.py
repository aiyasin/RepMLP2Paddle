from repmlp import Identity
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from repmlp import RepMLP
from repmlp import fuse_bn


class ConvBN(nn.Layer):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, groups=1, deploy=False, nonlinear=None):
        super().__init__()
        if nonlinear is None:
            self.nonlinear = Identity()
        else:
            self.nonlinear = nonlinear
        if deploy:
            self.conv = nn.Conv2D(in_channels=in_channels, out_channels=\
                out_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, groups=groups, bias_attr=True)
        else:
            self.conv = nn.Conv2D(in_channels=in_channels, out_channels=\
                out_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, groups=groups, bias_attr=False)
            self.bn = nn.BatchNorm2D(num_features=out_channels)

    def forward(self, x):
        if hasattr(self, 'bn'):
            return self.nonlinear(self.bn(self.conv(x)))
        else:
            return self.nonlinear(self.conv(x))

    def switch_to_deploy(self):
        kernel, bias = fuse_bn(self.conv, self.bn)
        conv = nn.Conv2D(in_channels=self.conv.in_channels, out_channels=\
            self.conv.out_channels, kernel_size=self.conv.kernel_size,
            stride=self.conv.stride, padding=self.conv.padding, groups=self
            .conv.groups, bias_attr=True)
        conv.weight.data = kernel
        conv.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('bn')
        self.conv = conv


class ConvBNReLU(ConvBN):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, groups=1, deploy=False):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, groups
            =groups, deploy=deploy, nonlinear=nn.ReLU())


class RepMLPLightBlock(nn.Layer):

    def __init__(self, in_channels, mid_channels, out_channels, H, W, h, w,
        reparam_conv_k, fc1_fc2_reduction, fc3_groups, deploy=False):
        super(RepMLPLightBlock, self).__init__()
        if in_channels != out_channels:
            self.shortcut = ConvBN(in_channels, out_channels, kernel_size=1,
                deploy=deploy)
        else:
            self.shortcut = Identity()
        self.light_conv1 = ConvBNReLU(in_channels, mid_channels,
            kernel_size=1, deploy=deploy)
        self.light_repmlp = RepMLP(in_channels=mid_channels, out_channels=\
            mid_channels, H=H, W=W, h=h, w=w, reparam_conv_k=reparam_conv_k,
            fc1_fc2_reduction=fc1_fc2_reduction, fc3_groups=fc3_groups,
            deploy=deploy)
        self.repmlp_nonlinear = nn.ReLU()
        self.light_conv3 = ConvBN(mid_channels, out_channels, kernel_size=1,
            deploy=deploy)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.light_conv1(x)
        out = self.light_repmlp(out)
        out = self.repmlp_nonlinear(out)
        out = self.light_conv3(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class RepMLPBottleneckBlock(nn.Layer):

    def __init__(self, in_channels, mid_channels, out_channels, r, H, W, h,
        w, reparam_conv_k, fc1_fc2_reduction, fc3_groups, deploy=False):
        super(RepMLPBottleneckBlock, self).__init__()
        if in_channels != out_channels:
            self.shortcut = ConvBN(in_channels, out_channels, kernel_size=1,
                deploy=deploy)
        else:
            self.shortcut = Identity()
        repmlp_channels = mid_channels // r
        self.btnk_conv1 = ConvBNReLU(in_channels, mid_channels, kernel_size
            =1, deploy=deploy)
        self.btnk_conv2 = ConvBNReLU(mid_channels, repmlp_channels,
            kernel_size=3, padding=1, deploy=deploy)
        self.btnk_repmlp = RepMLP(in_channels=repmlp_channels, out_channels
            =repmlp_channels, H=H, W=W, h=h, w=w, reparam_conv_k=\
            reparam_conv_k, fc1_fc2_reduction=fc1_fc2_reduction, fc3_groups
            =fc3_groups, deploy=deploy)
        self.repmlp_nonlinear = nn.ReLU()
        self.btnk_conv4 = ConvBNReLU(repmlp_channels, mid_channels,
            kernel_size=3, padding=1, deploy=deploy)
        self.btnk_conv5 = ConvBN(mid_channels, out_channels, kernel_size=1,
            deploy=deploy)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.btnk_conv1(x)
        out = self.btnk_conv2(out)
        out = self.btnk_repmlp(out)
        out = self.repmlp_nonlinear(out)
        out = self.btnk_conv4(out)
        out = self.btnk_conv5(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out
