import torch
from capsules.helpers import calc_same_padding
from capsules.layers.switchnorm import SwitchNorm3d
from torch import nn


def create_conv(in_channels,
                out_channels,
                kernel_size,
                order,
                num_groups,
                padding,
                dilation=1):
    """
    Create a list of modules with together constitute a single conv layer with
    non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of
        the input
        dilation (int or tuple): Dilation factor to create dilated conv, and
        increase the recceptive field

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[
        0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(
                ('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv',
                            nn.conv3d(in_channels,
                                      out_channels,
                                      kernel_size,
                                      bias,
                                      padding=padding,
                                      dilation=dilation)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than
            # the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm',
                            nn.GroupNorm(num_groups=num_groups,
                                         num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))

        elif char == 's':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(
                    ('SwitchNorm3d', SwitchNorm3d(in_channels, using_bn=False)))
            else:
                modules.append(
                    ('SwitchNorm3d', SwitchNorm3d(out_channels,
                                                  using_bn=False)))

        elif char == 'i':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(
                    ('InstanceNorm3d', nn.InstanceNorm3d(in_channels)))
            else:
                modules.append(
                    ('InstanceNorm3d', nn.InstanceNorm3d(out_channels)))

        else:
            raise ValueError(
                f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c', 's', 'i']"
            )

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and
    optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of
        the input
        dilation (int or tuple): Dilation factor to create dilated conv, and
        increase the recceptive field
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 order='gcr',
                 num_groups=8,
                 padding=1,
                 dilation=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels,
                                        out_channels,
                                        kernel_size,
                                        order,
                                        num_groups,
                                        padding=padding,
                                        dilation=dilation):
            self.add_module(name, module)


class ReshapeStem(nn.Module):

    def __init__(self, dim=1):
        super(ReshapeStem, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class StemCaps(nn.Sequential):

    def __init__(self,
                 input_size=64,
                 stem_size=1,
                 in_channels=1,
                 stem_channels=128,
                 stem_kernel_size=5,
                 stem_order="cr",
                 stem_dilation=1,
                 reshape=True):
        super(StemCaps, self).__init__()
        assert stem_size > 0

        if type(stem_channels) != list:
            stem_channels = [stem_channels] * stem_size
        if type(stem_kernel_size) != list:
            stem_kernel_size = [stem_kernel_size] * stem_size
        if type(stem_order) != list:
            stem_order = [stem_order] * stem_size
        if type(stem_dilation) != list:
            stem_dilation = [stem_dilation] * stem_size

        in_channels = [in_channels] + stem_channels[:-1]

        # Safety checks
        should_match_stem_size = [
            in_channels, stem_channels, stem_kernel_size, stem_order,
            stem_dilation
        ]
        for var in should_match_stem_size:
            assert len(
                var
            ) == stem_size, f"Incorrect list: {var}. Its length should match `stem_size`."

        # Adding a succession of convolutions with `same` padding
        for stem_conv in range(stem_size):
            p, _ = calc_same_padding(
                input_size,
                kernel=stem_kernel_size[stem_conv],
                dilation=stem_dilation[stem_conv],
            )
            self.add_module(
                f"StemConv{stem_conv}",
                SingleConv(in_channels[stem_conv],
                           stem_channels[stem_conv],
                           kernel_size=stem_kernel_size[stem_conv],
                           order=stem_order[stem_conv],
                           num_groups=8,
                           padding=p,
                           dilation=stem_dilation[stem_conv]))

        if reshape:
            self.add_module("ReshapeStem", ReshapeStem())


class StemBlock(nn.Module):

    def __init__(self,
                 input_size=64,
                 n_caps=4,
                 stem_size=3,
                 in_channels=1,
                 stem_channels=[16, 16, 16],
                 stem_kernel_sizes=[[1, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1]],
                 stem_order="cr",
                 stem_dilations=[[1, 1, 1], [6, 1, 1], [12, 1, 1], [18, 1, 1]]):
        super(StemBlock, self).__init__()

        stems = [
            StemCaps(input_size=input_size,
                     stem_size=stem_size,
                     in_channels=in_channels,
                     stem_channels=stem_channels,
                     stem_kernel_size=stem_kernel_sizes[c],
                     stem_order=stem_order,
                     stem_dilation=stem_dilations[c],
                     reshape=False) for c in range(n_caps)
        ]
        self.stems = nn.ModuleList(stems)

    def forward(self, x):
        x = torch.stack([stem(x) for stem in self.stems], dim=1)
        return x
