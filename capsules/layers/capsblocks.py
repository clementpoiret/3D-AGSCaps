import torch
from capsules.helpers import calc_same_padding, squash
from capsules.layers.attention import AttentionBlock
from capsules.layers.convcaps import ConvCapsuleLayer3D
from torch import nn


class SingleCaps(nn.Sequential):
    """
    Basic capsule module
    """

    def __init__(self,
                 kernel_size,
                 input_num_capsule,
                 input_num_atoms,
                 num_capsule,
                 num_atoms,
                 strides=1,
                 routings=3,
                 activation=squash,
                 sigmoid_routing=True,
                 transposed=False,
                 constrained=True,
                 final_squash=True,
                 use_switchnorm=False):
        super(SingleCaps, self).__init__()
        o = 64
        k = kernel_size - 1 if transposed else kernel_size
        p, _ = calc_same_padding(input_=o,
                                 kernel=k,
                                 stride=strides,
                                 transposed=transposed)
        self.add_module(
            "Capsule",
            ConvCapsuleLayer3D(kernel_size=k,
                               input_num_capsule=input_num_capsule,
                               input_num_atoms=input_num_atoms,
                               num_capsule=num_capsule,
                               num_atoms=num_atoms,
                               strides=strides,
                               padding=p,
                               routings=routings,
                               sigmoid_routing=sigmoid_routing,
                               transposed=transposed,
                               constrained=constrained,
                               activation=activation,
                               final_squash=final_squash,
                               use_switchnorm=use_switchnorm))


class DoubleCaps(nn.Sequential):
    """
    A module consisting of two consecutive capsule layers
    """

    def __init__(self,
                 kernel_size,
                 input_num_capsule,
                 input_num_atoms,
                 num_capsule,
                 num_atoms,
                 strides=1,
                 routings=3,
                 activation=squash,
                 sigmoid_routing=True,
                 transposed=False,
                 constrained=True,
                 final_squash=True,
                 use_switchnorm=False):
        super(DoubleCaps, self).__init__()

        if type(strides) != list:
            strides = [strides, strides]
        if type(transposed) != list:
            transposed = [transposed, transposed]
        if type(routings) != list:
            routings = [routings, routings]
        if type(activation) != list:
            activation = [activation, activation]
        if type(final_squash) != list:
            final_squash = [final_squash, final_squash]
        if type(use_switchnorm) != list:
            use_switchnorm = [use_switchnorm, use_switchnorm]
        input_capsules = [input_num_capsule, num_capsule]
        input_atoms = [input_num_atoms, num_atoms]

        for i, stride in enumerate(strides):
            o = 64
            k = kernel_size - 1 if transposed[i] else kernel_size
            p, _ = calc_same_padding(input_=o,
                                     kernel=k,
                                     stride=stride,
                                     transposed=transposed[i])
            self.add_module(
                f'Capsule{i}',
                ConvCapsuleLayer3D(kernel_size=k,
                                   input_num_capsule=input_capsules[i],
                                   input_num_atoms=input_atoms[i],
                                   num_capsule=num_capsule,
                                   num_atoms=num_atoms,
                                   strides=stride,
                                   padding=p,
                                   routings=routings[i],
                                   sigmoid_routing=sigmoid_routing,
                                   transposed=transposed[i],
                                   constrained=constrained,
                                   activation=activation[i],
                                   final_squash=final_squash[i],
                                   use_switchnorm=use_switchnorm[i]))


class AttentionDecoderCaps(nn.Module):
    """
    A single module for decoder path consisting of the upsampling caps
    followed by a basic capsule. Using encoder_features as attention gate
    Args:
        todo.
    """

    def __init__(self,
                 kernel_size,
                 input_num_capsule,
                 input_num_atoms,
                 num_capsule,
                 num_atoms,
                 strides=[2, 1],
                 routings=1,
                 sigmoid_routing=True,
                 transposed=[True, False],
                 constrained=True,
                 union_type=None,
                 normalization="s",
                 using_bn=False):
        super(AttentionDecoderCaps, self).__init__()
        self.union_type = union_type
        if isinstance(routings, int):
            routings = [routings] * 2

        self.transposed_caps = SingleCaps(kernel_size=kernel_size,
                                          input_num_capsule=input_num_capsule,
                                          input_num_atoms=input_num_atoms,
                                          num_capsule=num_capsule,
                                          num_atoms=num_atoms,
                                          strides=strides[0],
                                          routings=routings[0],
                                          sigmoid_routing=sigmoid_routing,
                                          transposed=transposed[0],
                                          constrained=constrained)
        _f = 2 if union_type == "cat" else 1
        self.caps = SingleCaps(kernel_size=kernel_size,
                               input_num_capsule=num_capsule * _f,
                               input_num_atoms=num_atoms,
                               num_capsule=num_capsule,
                               num_atoms=num_atoms,
                               strides=strides[1],
                               routings=routings[1],
                               sigmoid_routing=sigmoid_routing,
                               transposed=transposed[1],
                               constrained=constrained)

        self.att = AttentionBlock(F_g=num_capsule * num_atoms,
                                  F_l=num_capsule * num_atoms,
                                  F_int=(num_capsule * num_atoms) // 2,
                                  F_out=num_atoms,
                                  normalization=normalization,
                                  using_bn=using_bn)

    def forward(self, encoder_features, x):
        up = self.transposed_caps(x)

        att = self.att(encoder_features, up)

        if self.union_type == "cat":
            x = torch.cat((att, up), dim=1)
        elif self.union_type == "sum":
            x = up + att
            # todo: squash?
        else:
            x = att
        x = self.caps(x)

        return x


class EncoderCaps(nn.Module):
    """
    A single module from the encoder path consisting of a double capsule block.
    The first capsule may have a stride of 2, and the second a stride of 1, so
    that spatial dim is divided by 2.
    Args:
        todo.
    """

    def __init__(self,
                 kernel_size,
                 input_num_capsule,
                 input_num_atoms,
                 num_capsule,
                 num_atoms,
                 basic_module=DoubleCaps,
                 strides=[2, 1],
                 routings=1,
                 sigmoid_routing=True,
                 transposed=False,
                 constrained=True,
                 use_switchnorm=False):
        super(EncoderCaps, self).__init__()

        self.basic_module = basic_module(kernel_size=kernel_size,
                                         input_num_capsule=input_num_capsule,
                                         input_num_atoms=input_num_atoms,
                                         num_capsule=num_capsule,
                                         num_atoms=num_atoms,
                                         strides=strides,
                                         routings=routings,
                                         sigmoid_routing=sigmoid_routing,
                                         transposed=transposed,
                                         constrained=constrained,
                                         use_switchnorm=use_switchnorm)

    def forward(self, x):
        x = self.basic_module(x)
        return x


class DecoderCaps(nn.Module):
    """
    A single module for decoder path consisting of the upsampling caps
    followed by a basic capsule.
    Args:
        todo.
    """

    def __init__(self,
                 kernel_size,
                 input_num_capsule,
                 input_num_atoms,
                 num_capsule,
                 num_atoms,
                 strides=[2, 1],
                 routings=1,
                 sigmoid_routing=True,
                 transposed=[True, False],
                 constrained=True,
                 use_switchnorm=False,
                 union_type="cat"):
        super(DecoderCaps, self).__init__()
        self.union_type = union_type
        if isinstance(routings, int):
            routings = [routings] * 2

        self.transposed_caps = SingleCaps(kernel_size=kernel_size,
                                          input_num_capsule=input_num_capsule,
                                          input_num_atoms=input_num_atoms,
                                          num_capsule=num_capsule,
                                          num_atoms=num_atoms,
                                          strides=strides[0],
                                          routings=routings[0],
                                          sigmoid_routing=sigmoid_routing,
                                          transposed=transposed[0],
                                          constrained=constrained,
                                          use_switchnorm=use_switchnorm)
        _f = 2 if union_type == "cat" else 1
        self.caps = SingleCaps(kernel_size=kernel_size,
                               input_num_capsule=num_capsule * _f,
                               input_num_atoms=num_atoms,
                               num_capsule=num_capsule,
                               num_atoms=num_atoms,
                               strides=strides[1],
                               routings=routings[1],
                               sigmoid_routing=sigmoid_routing,
                               transposed=transposed[1],
                               constrained=constrained,
                               use_switchnorm=use_switchnorm)

    def forward(self, encoder_features, x):
        up = self.transposed_caps(x)
        if self.union_type == "cat":
            x = torch.cat((encoder_features, up), dim=1)
        elif self.union_type == "sum":
            x = up + encoder_features
            # todo: squash?
        x = self.caps(x)

        return x