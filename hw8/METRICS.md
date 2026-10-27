# Homework 8 Metrics

## Configuration

Dataset: MNIST

Diffusion steps: 100

Beta schedule: linear from 0.0001 to 0.02

Batch size: 128

Training epochs: 50

Optimizer: Adam

Learning rate: 0.001

Loss function: Mean Squared Error

Device: NVIDIA Tesla T4

## Baseline DDPM Model

Trainable parameters: 82,945

Initial epoch loss: 0.156066

Final epoch loss: 0.062526

Best epoch loss: 0.062490

Best epoch: 48

Loss reduction: 59.94%

Average epoch time: 12.71 seconds

Total training time: 635.28 seconds

Sampling evaluation showed poor global digit structure despite the low noise prediction loss.

## Improved DDPM Model

Trainable parameters: 517,665

Initial epoch loss: 0.181950

Final epoch loss: 0.060271

Best epoch loss: 0.060216

Best epoch: 45

Loss reduction: 66.88%

Total training time: 714.68 seconds

The improved architecture produced more spatially coherent digit like samples than the baseline model.

## Sampling Performance

DDPM 100 steps: 0.2366 seconds

DDIM 50 steps: 0.1354 seconds

DDIM 20 steps: 0.0499 seconds

DDIM 50 speedup over DDPM: 1.75x

DDIM 20 speedup over DDPM: 4.74x

DDIM 50 sampling time reduction: 42.77%

DDIM 20 sampling time reduction: 78.89%

## Qualitative Evaluation

The 100 step DDPM sampler produced the most coherent generated structures overall.

DDIM with 50 steps reduced generation time while retaining some recognizable digit like structure.

DDIM with 20 steps provided the largest speed improvement but generated more distorted and ambiguous samples.

The experiment demonstrates a clear tradeoff between sampling speed and qualitative generation quality.
