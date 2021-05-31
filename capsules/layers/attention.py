from torch import nn

from .switchnorm import SwitchNorm3d


class AttentionBlock(nn.Module):
    """
    3D Caps Attention Block w/ optional Normalization.
    For normalization, it supports:
    - `b` for `BatchNorm3d`,
    - `s` for `SwitchNorm3d`.
    
    `using_bn` controls SwitchNorm's behavior. It has no effect is
    `normalization == "b"`.

    SwitchNorm3d comes from:
    <https://github.com/switchablenorms/Switchable-Normalization>
    """

    def __init__(self,
                 F_g,
                 F_l,
                 F_int,
                 F_out=1,
                 normalization=None,
                 using_bn=False):
        super(AttentionBlock, self).__init__()

        W_g = [
            nn.Conv3d(
                F_g,
                F_int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        ]
        W_x = [
            nn.Conv3d(
                F_l,
                F_int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        ]
        psi = [
            nn.Conv3d(
                F_int,
                F_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        ]
        if normalization == "b":
            W_g.append(nn.BatchNorm3d(F_int))
            W_x.append(nn.BatchNorm3d(F_int))
            psi.append(nn.BatchNorm3d(F_out))
        elif normalization == "s":
            W_g.append(SwitchNorm3d(F_int, using_bn=using_bn))
            W_x.append(SwitchNorm3d(F_int, using_bn=using_bn))
            psi.append(SwitchNorm3d(F_out, using_bn=using_bn))

        self.W_g = nn.Sequential(*W_g)
        self.W_x = nn.Sequential(*W_x)

        psi.append(nn.Sigmoid())
        self.psi = nn.Sequential(*psi)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # Reshaping
        # g & x should normally have the same shape here
        # I don't think we should be more specific right now.
        bs, C, A, a, b, c = g.shape

        g1 = self.W_g(g.view(bs, C * A, a, b, c))
        x1 = self.W_x(x.view(bs, C * A, a, b, c))
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Unsqueeze to match capsule dimension
        psi = psi.unsqueeze(1)
        out = x * psi

        return out
