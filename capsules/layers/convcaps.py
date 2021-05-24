import capsules.helpers as h
import torch
from capsules.layers.switchnorm import SwitchNorm3d
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class Length(nn.Module):

    def __init__(self, dim=1, keepdim=True, p='fro'):
        super(Length, self).__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.p = p

    def forward(self, inputs):
        return inputs.norm(dim=self.dim, keepdim=self.keepdim, p=self.p)


class SafeLength(nn.Module):

    def __init__(self, dim=2, keepdim=False, eps=1e-7):
        super(SafeLength, self).__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.eps = eps

    def forward(self, x):
        squared_norm = torch.sum(torch.square(x),
                                 axis=self.dim,
                                 keepdim=self.keepdim)
        return torch.sqrt(squared_norm + self.eps)


class ConvCapsuleLayer3D(nn.Module):
    # Should I implement leaky routing?
    # This would allow orphans when Softmax routing
    def __init__(self,
                 kernel_size,
                 input_num_capsule,
                 input_num_atoms,
                 num_capsule,
                 num_atoms,
                 strides=1,
                 padding=0,
                 routings=3,
                 activation=h.squash,
                 final_squash=True,
                 sigmoid_routing=True,
                 transposed=False,
                 constrained=False,
                 use_switchnorm=False,
                 use_batchnorm=False):
        super(ConvCapsuleLayer3D, self).__init__()
        self.input_num_capsule = input_num_capsule
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.strides = strides
        self.padding = padding
        self.routings = routings
        self.activation = activation
        self.final_squash = final_squash
        self.sigmoid_routing = sigmoid_routing
        self.constrained = constrained

        in_channels = input_num_atoms if constrained else input_num_capsule * input_num_atoms
        out_channels = num_capsule * num_atoms if constrained else input_num_capsule * num_capsule * num_atoms
        if transposed:
            self.conv = nn.ConvTranspose3d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           stride=strides,
                                           padding=padding,
                                           bias=False)
        else:
            self.conv = nn.Conv3d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=strides,
                                  padding=padding,
                                  bias=False)

        # nn.init.normal_(self.conv.weight.data, mean=0., std=1.)
        self.conv.weight.data = nn.init.kaiming_normal_(self.conv.weight.data)
        # nn.init.xavier_uniform_(self.conv.weight.data)

        if use_switchnorm:
            self.conv = nn.Sequential(
                self.conv, SwitchNorm3d(out_channels, using_bn=use_batchnorm))

        self.b = nn.Parameter(torch.full((num_capsule, num_atoms, 1, 1, 1),
                                         0.1))

    def forward(self, input_tensor):
        # input shape -> (bs, C, A, x, y, z)
        input_shape = input_tensor.shape

        if self.constrained:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0] * input_shape[1], input_shape[2], input_shape[3],
                input_shape[4], input_shape[5])  # -> (bs*C,A,x,y,z)
        else:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0], input_shape[1] * input_shape[2], input_shape[3],
                input_shape[4], input_shape[5])  # -> (bs,C*A,x,y,z)

        conv = self.conv(input_tensor_reshaped)

        # votes_shape = conv.shape
        _, _, conv_height, conv_width, conv_depth = conv.shape

        votes = conv.view(input_shape[0], self.input_num_capsule,
                          self.num_capsule, self.num_atoms, conv_height,
                          conv_width, conv_depth)  # -> (bs,input_C,C,A,x,y,z)

        # votes: (bs,input_C,x,y,z,C,A)
        # votes = votes.permute(0, 1, 4, 5, 6, 2, 3)

        logit_shape = torch.Size([
            input_shape[0],
            self.input_num_capsule,
            self.num_capsule,
            conv_height,
            conv_width,
            conv_depth,
        ])  # -> (bs,input_C,x,y,z,C)
        biases_replicated = self.b.repeat(1, 1, conv_height, conv_width,
                                          conv_depth)  # -> (x,y,z,C,A)

        activations = update_routing(
            votes=votes,
            biases=biases_replicated,
            logit_shape=logit_shape,
            num_dims=7,
            num_routing=self.routings,
            final_nonlinearity=self.activation,
            final_squash=self.final_squash,
            sigmoid_routing=self.sigmoid_routing)  # -> (bs,x,y,z,C,A)

        return activations  # -> (bs,C,A,x,y,z)


def update_routing(votes, biases, logit_shape, num_dims, num_routing,
                   final_nonlinearity, final_squash, sigmoid_routing):
    # votes: (bs,1,128,128,128,8,32)
    if num_dims == 7:
        votes_t_shape = 3, 0, 1, 2, 4, 5, 6
        r_t_shape = 1, 2, 3, 0, 4, 5, 6
    else:
        raise NotImplementedError('Not implemented')

    votes_trans = votes.permute(votes_t_shape)  # -> (A,bs,input_C,x,y,z,C)
    # _, _, _, height, width, depth, caps = votes_trans.shape

    # activations = []

    if sigmoid_routing:
        routing = torch.sigmoid
        logits = Variable(torch.ones(logit_shape), requires_grad=False).to(
            votes.device)  # -> (bs,input_C,x,y,z,C)
    else:
        routing = nn.Softmax(dim=2)
        logits = Variable(torch.zeros(logit_shape), requires_grad=False).to(
            votes.device)  # -> (bs,input_C,x,y,z,C)

    for i in range(num_routing):
        """Routing while loop."""
        route = routing(logits)  # -> (bs,input_C,x,y,z,C)

        if i == num_routing - 1:
            preactivate_unrolled = route * votes_trans  # -> (A,bs,input_C,x,y,z,C)
            preact_trans = preactivate_unrolled.permute(
                r_t_shape)  # -> (bs,input_C,x,y,z,C,A)
            preactivate = preact_trans.sum(dim=1) + biases  # -> (bs,x,y,z,C,A)
            if final_squash:
                activation = final_nonlinearity(
                    preactivate, dim=2, caps_dim=1,
                    atoms_dim=2)  # -> (bs,x,y,z,C,A)
            else:
                activation = preactivate

        else:
            preactivate_unrolled = route * votes_trans  # -> (A,bs,input_C,x,y,z,C)
            preact_trans = preactivate_unrolled.permute(
                r_t_shape)  # -> (bs,input_C,x,y,z,C,A)
            preactivate = preact_trans.sum(dim=1) + biases  # -> (bs,x,y,z,C,A)
            activation = h.squash(preactivate, safe=True,
                                  dim=2)  # -> (bs,x,y,z,C,A)

            act_3d = activation.data.unsqueeze(1)  # -> (bs,1,x,y,z,C,A)
            distances = torch.sum(votes * act_3d, dim=3)
            logits += distances

    return activation
