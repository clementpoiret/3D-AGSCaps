from functools import partial

import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from torch import optim

from agscaps.data import SyntheticDataModule
from agscaps.loss import FocalTversky_loss
from agscaps.models import SegmentationModel

SEED = 42
N_CLASSES = 3
HPARAMS = {
    "α": 1,  # The weight of the reconstruction loss
    "in_channels": 1,
    "out_channels": N_CLASSES,
    "n_caps": 4,  # Equals the number of branches in the ASFE
    "n_atoms": [64, 16, 16, 8],  # Defines the number of caps per level
    "num_levels": 4,  # Depth of the U-Net architecture
    "stem_order": "crs",  # Conv -> ReLU -> SwitchNorm in the ASFE
    "stem_channels": [128, 64, 64],  # Number of channels in ASFE's convs
    "stem_size": 3,  # N of convs in ASFE
    "stem_kernel_size": [[1, 1, 1], [5, 3, 1], [5, 3, 1], [5, 3,
                                                           1]],  # ASFE's ks
    "stem_dilations": [[1, 1, 1], [1, 1, 1], [2, 1, 1],
                       [4, 1, 1]],  # ASFE's dilations
    "segcaps_num_atoms":
        64,  # Dimensionality of the vectors in the last capsule performing segmentation
    "segcaps_routings":
        1,  # Routing by agreement iterations in the last capsule
    "segcaps_use_switchnorm": True,  # SwitchNorm before taking vectors' norms
    "use_switchnorm": True,  # SwitchNorm after capsules, except the last one
    "union_type":
        None  # Union between encoder and decoder, with attention. "cat", "sum", or None
}


def main():
    seed_everything(SEED)

    # Generating a synthetic dataset of size (bs,1,32,32,32)
    datamodule = SyntheticDataModule(n_train=128,
                                     n_val=16,
                                     n_test=16,
                                     size=(32, 32, 32),
                                     n_classes=N_CLASSES,
                                     batch_size=1,
                                     pin_memory=True,
                                     num_workers=8)

    # Creating an AGSCaps model
    model = SegmentationModel(HPARAMS,
                              seg_loss=FocalTversky_loss({
                                  "apply_nonlin": None,
                              }),
                              rec_loss=F.mse_loss,
                              optimizer=optim.AdamW,
                              scheduler=partial(
                                  optim.lr_scheduler.CosineAnnealingLR,
                                  T_max=32),
                              learning_rate=1e-3)

    # PyTorch Lightning Trainer
    trainer = Trainer(gpus=1,
                      precision=16,
                      log_gpu_memory=True,
                      max_epochs=32,
                      progress_bar_refresh_rate=8,
                      benchmark=True,
                      deterministic=True,
                      stochastic_weight_avg=True,
                      checkpoint_callback=False)

    # Training & Validating
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
