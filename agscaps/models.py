import pytorch_lightning as pl
import torch.nn.functional as F
from einops import rearrange
from torch import nn, optim

from .helpers import number_of_features_per_level, smsquash, squash
from .layers.asfe import StemBlock
from .layers.capsblocks import AttentionDecoderCaps, DoubleCaps, EncoderCaps
from .layers.convcaps import Length
from .loss import FocalTversky_loss


class AGSCaps(nn.Module):
    """
    Base class for 3D-AGSCaps.

    Args:
        todo.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_caps=1,
                 n_atoms=16,
                 num_levels=4,
                 conv_kernel_size=3,
                 stem_size=3,
                 stem_channels=16,
                 stem_kernel_size=[[1, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1]],
                 stem_dilations=[[1, 1, 1], [6, 1, 1], [12, 1, 1], [18, 1, 1]],
                 stem_order="cr",
                 constrained=True,
                 use_switchnorm=False,
                 segcaps_module=DoubleCaps,
                 segcaps_kernel_size=1,
                 segcaps_num_atoms=32,
                 segcaps_strides=1,
                 segcaps_routings=3,
                 segcaps_sigmoid_routings=True,
                 segcaps_constrained=True,
                 segcaps_use_switchnorm=False,
                 final_rec_sigmoid=False,
                 normalization="s",
                 using_bn=False,
                 union_type=None,
                 **kwargs):
        super(AGSCaps, self).__init__()

        if isinstance(n_caps, int):
            n_caps = number_of_features_per_level(n_caps, num_levels=num_levels)
        if isinstance(n_atoms, int):
            n_atoms = number_of_features_per_level(n_atoms,
                                                   num_levels=num_levels)

        # create encoder path consisting of Encoder modules.
        # Depth of the encoder is equal to `len(n_caps)`
        encoders = []
        for i, out_caps_num in enumerate(n_caps):
            if i == 0:
                encoder = StemBlock(n_caps=n_caps[i],
                                    stem_size=stem_size,
                                    in_channels=in_channels,
                                    stem_channels=stem_channels,
                                    stem_kernel_sizes=stem_kernel_size,
                                    stem_order=stem_order,
                                    stem_dilations=stem_dilations)
            else:
                encoder = EncoderCaps(kernel_size=conv_kernel_size,
                                      input_num_capsule=n_caps[i - 1],
                                      input_num_atoms=n_atoms[i - 1],
                                      num_capsule=out_caps_num,
                                      num_atoms=n_atoms[i],
                                      strides=[2, 1],
                                      routings=1,
                                      constrained=constrained,
                                      use_switchnorm=use_switchnorm)
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the DecoderCaps modules.
        # The length of the decoder is equal to `len(n_caps) - 1`
        decoders = []
        reversed_n_caps = list(reversed(n_caps))
        reversed_n_atoms = list(reversed(n_atoms))
        # in_caps_nums = list(map(add, reversed_n_caps, ([0] + reversed_n_caps[:-1])))
        for i in range(len(reversed_n_caps) - 1):
            # todo: this is for "cat" union
            in_caps_num = reversed_n_caps[i]
            in_atoms_num = reversed_n_atoms[i]
            out_caps_num = reversed_n_caps[i + 1]
            out_atoms_num = reversed_n_atoms[i + 1]

            decoder = AttentionDecoderCaps(kernel_size=conv_kernel_size,
                                           input_num_capsule=in_caps_num,
                                           input_num_atoms=in_atoms_num,
                                           num_capsule=out_caps_num,
                                           num_atoms=out_atoms_num,
                                           strides=[2, 1],
                                           routings=1,
                                           sigmoid_routing=True,
                                           transposed=[True, False],
                                           constrained=constrained,
                                           union_type=union_type,
                                           normalization=normalization,
                                           using_bn=using_bn)
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1x1 capsule reduces the number of output
        # channels to the number of labels
        self.segcaps = segcaps_module(kernel_size=segcaps_kernel_size,
                                      input_num_capsule=n_caps[0],
                                      input_num_atoms=n_atoms[0],
                                      num_capsule=out_channels,
                                      num_atoms=segcaps_num_atoms,
                                      strides=segcaps_strides,
                                      routings=segcaps_routings,
                                      sigmoid_routing=segcaps_sigmoid_routings,
                                      constrained=segcaps_constrained,
                                      use_switchnorm=segcaps_use_switchnorm,
                                      activation=squash,
                                      final_squash=False)

        # Length layer to output capsules' predictions
        self.outseg = Length(dim=2, keepdim=False, p=2)

        # Reconstruction
        reconstructor = [
            nn.Conv3d(in_channels=segcaps_num_atoms * out_channels,
                      out_channels=64,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64,
                      out_channels=64,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64,
                      out_channels=1,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        ]
        if final_rec_sigmoid:
            reconstructor.append(nn.Sigmoid())
        self.reconstructor = nn.Sequential(*reconstructor)

    def forward(self, x, y=None):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        segcaps = self.segcaps(x)

        # Computes actual segmentation with vectors' norm
        segmentation = self.outseg(smsquash(segcaps))

        # For training, the true label is used to mask the output of
        # capsule layer.
        # For prediction, mask using the capsule with maximal length.
        if y is not None:
            # Creating a mask (discarding the background class)
            _, mask = y.max(dim=1, keepdim=True)
            mask = mask.clip(0, 1)
        else:
            # Taking the length of the vector as voxels' classes
            classes = segcaps.norm(dim=2, p=2)
            _, mask = classes.max(dim=1, keepdim=True)

        mask = rearrange(mask, "b c h w d -> b c 1 h w d")
        masked = segcaps * mask

        # Merging caps and atoms
        masked = rearrange(masked, "b c a h w d -> b (c a) h w d")

        # Reconstructing via the reconstructor network
        reconstruction = self.reconstructor(masked)

        return segmentation, reconstruction


class SegmentationModel(pl.LightningModule):

    def __init__(self,
                 hparams,
                 seg_loss=FocalTversky_loss({
                     "apply_nonlin": nn.Softmax(dim=1),
                 }),
                 rec_loss=F.mse_loss,
                 optimizer=optim.AdamW,
                 scheduler=None,
                 learning_rate=1e-3,
                 classes_names=None,
                 is_capsnet=False):
        super(SegmentationModel, self).__init__()
        self.α = hparams['α']
        self.learning_rate = learning_rate
        self.seg_loss = seg_loss
        self.rec_loss = rec_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.is_capsnet = is_capsnet
        if classes_names:
            assert len(classes_names) == hparams['out_channels']
        self.classes_names = classes_names

        self._model = AGSCaps(**hparams)

    def forward(self, x, y=None):
        return self._model(x, y)

    def default_step(self, batch, log_preffix="Training"):
        x, y = batch
        _, labels = y.max(dim=1)

        y_hat, reconstruction = self.forward(x, y)
        mask = labels.unsqueeze(1).clip(0, 1)
        target = x * mask
        rec_loss = self.rec_loss(reconstruction, target)
        seg_loss = self.seg_loss(y_hat, labels.long())
        loss = seg_loss + self.α * rec_loss

        self.log(f"{log_preffix} SegLoss", seg_loss)
        self.log(f"{log_preffix} RecLoss", rec_loss)

        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        optimizers = {"optimizer": optimizer}

        if self.scheduler:
            optimizers["lr_scheduler"] = self.scheduler(optimizer)
            optimizers["monitor"] = "Validation SegLoss"

        return optimizers

    def training_step(self, batch, batch_idx):
        return self.default_step(batch, "Training")

    def validation_step(self, batch, batch_idx):
        return self.default_step(batch, "Validation")

    def test_step(self, batch, batch_idx):
        return self.default_step(batch, "Test")
