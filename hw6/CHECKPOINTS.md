# HW6 Model Checkpoints

The trained model checkpoints are stored externally because the combined checkpoint size is too large for a normal GitHub push.

Checkpoint archive:

https://drive.google.com/file/d/1dMe0Sen6QLa_8kW0uJAV-rdCA7qP23Vz/view?usp=sharing

The archive contains:

1. `supervised_resnet18.pt`
2. `rotation_resnet18.pt`
3. `rotation_linear_classifier.pt`
4. `simclr_encoder.pt`
5. `simclr_linear_classifier.pt`

The models were trained by running `HW6.ipynb`. The notebook contains the complete training configuration, seeds, dataset subsets, training loops, and reported results.
