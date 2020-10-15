Defending against black-box adversarial attacks with gradient-free trained sign activation neural networks
=

Guide
-
Directory
-  `binary` Contains training scripts for scd with different variations, MLP-logistic,
Binarized Neural Network (bnn), CNN models (ResNet, LeNet). Substitute model attack scripts.
HopSkipJump attack scripts.

-  `core` Contains scd01mlp, scd01mlp-bnn, scdcemlp, scdcdmlp-bnn implementation.

- `tools` Contain Args definition, data sets preparing and loading interface, and other
tools will be used in the training scripts.
 

Python Environment
-

python v3.6.8

pytorch v1.3.1

numpy v1.17.4

pandas v1.0.3

argparse v1.1

sklearn v0.21.2

Larq v0.9 (https://github.com/larq/larq)

skorch v0.8.0

ART v1.4.0 (https://github.com/Trusted-AI/adversarial-robustness-toolbox)

