3D-AGSCaps
========

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5506955.svg
   :target: https://doi.org/10.5281/zenodo.5506955

3D-AGSCaps is mainly provided as a proof-of-concept. We wanted to show that Capsule Networks (ref) were able to accurately perform 3D segmentation tasks.

To assess this hypothesis, we tested our model on hippocampal segmentation (antero-posterior & hippocampal subfields).

- Original poster: https://hippomnesis.dev/2021/06/10/3d-agscaps/
- Paper: in prep.

Currently, we do not provide a pretrained model, but it is in our to-do list. In the meantime, you can use our layers/architectures like in the `main.py` file:

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

- 3D Capsule Layer,
- SMSquash activation function for Multiclass classification tasks,
- Atrous Spatial Feature Extraction for multi-scale feature extraction before entering the Capsule Network,
- Attention Gates to modulate Capsules' L2 norm,
- Basic UNet-like architecture.

What's next?
------------

- [ ] Provide pre-trained models,
- [ ] Publish a reference paper,
- [ ] Provide a working software to segment the hippocampus with a GUI (no code required) using pre-trained models.

Installation
------------

Install the latest 3D-AGSCaps 0.2.0 (refactoring using Einops) by running:

.. code-block::

    pip install https://github.com/clementpoiret/3D-AGSCaps/releases/download/v0.2.0/AGSCaps-0.2.0-py3-none-any.whl
    

Or, install 3D-AGSCaps 0.1.0 (original version of the paper / poster) by running:

.. code-block::

    pip install https://github.com/clementpoiret/3D-AGSCaps/releases/download/v0.1.0/AGSCaps-0.1.0-py3-none-any.whl

Contribute
----------

- Issues or suggestions? Feel free to open an issue or a pull request! :)

Support
-------

If you are having issues, please let us know at clement.poiret[at]cea.fr

License
-------

The project is licensed under the MIT license.

To cite this work, please see the following example:

``Clément POIRET. (2021). clementpoiret/3D-AGSCaps: Zenodo Release (v0.2.1). Zenodo. https://doi.org/10.5281/zenodo.5506955``
