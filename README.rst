3D-AGSCaps
========

3D-AGSCaps is mainly provided to serve as a proof-of-concept. We wanted to show that Capsule Networks (ref) were able to accurately perform 3D segmentation tasks.

To assess this hypothesis, we tested our model on hippocampal segmentation (antero-posterior & hippocampal subfields).

As of now, we do not provide a pretrained model, but it is in my to-do list. In the meantime, you can use our layers/architectures like in the `main.py` file:

.. code-block:: python

    from agscaps.models import SegmentationModel
    
    # Declare your global variables
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
    # Assuming your data is in a `datamodule` object
    trainer.fit(model, datamodule=datamodule)


Features
--------

- Be awesome
- Make things faster

Installation
------------

Install $project by running:

.. code-block::

    install project

Contribute
----------

- Issue Tracker: github.com/$project/$project/issues
- Source Code: github.com/$project/$project

Support
-------

If you are having issues, please let us know.
We have a mailing list located at: project@google-groups.com

License
-------

The project is licensed under the BSD license.
