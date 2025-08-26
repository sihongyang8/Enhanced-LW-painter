import torch.nn as nn
from models.ean import ean_module
import torch.nn.functional as F

class BSConvU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


class MSFA(nn.Module):
    '''Multi-scale feature aggregation.'''
    def __init__(self, in_channels=256, out_channels=256,dilation_rate_list=[1, 2, 4, 8]):
        super(MSFA, self).__init__()
        self.dilation_rate_list = dilation_rate_list

        for _, dilation_rate in enumerate(dilation_rate_list):

            self.__setattr__('dilated_conv_{:d}'.format(_), nn.Sequential(
                BSConvU(in_channels,out_channels,dilation=dilation_rate,padding=dilation_rate,kernel_size=3),
                nn.ReLU(inplace=True))
                             )

        self.weight_calc = nn.Sequential(
            BSConvU(in_channels, out_channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, len(dilation_rate_list), 1),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )

        self.conv=BSConvU(in_channels,out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):

        weight_map = self.weight_calc(x)

        x_feature_list = []
        for _, dilation_rate in enumerate(self.dilation_rate_list):
            x_feature_list.append(
                self.__getattr__('dilated_conv_{:d}'.format(_))(x)
            )

        output = weight_map[:, 0:1, :, :] * x_feature_list[0] + \
                 weight_map[:, 1:2, :, :] * x_feature_list[1] + \
                 weight_map[:, 2:3, :, :] * x_feature_list[2] + \
                 weight_map[:, 3:4, :, :] * x_feature_list[3]

        output2=self.conv(x)
        output=F.gelu(output2)*output
        output=self.conv3(output)

        return output



class EFFBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.ean = ean_module(channels=dim)
        self.msb = MSFA(in_channels=dim, out_channels=dim)

    def forward(self, x):
        identity = x
        x = self.ean(x) + identity
        identity2 = x
        x = self.norm(x)
        x = self.msb(x) + identity2

        return x

