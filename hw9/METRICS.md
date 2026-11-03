# HW9 Metrics

## Student
SID4: 2046
Seed: 2046

## Environment
GPU: Tesla T4
Dataset: FashionMNIST
Image resolution: 128 x 128

## VAE Compression
Pretrained VAE: stabilityai/sd-vae-ft-mse
Latent scaling factor: 0.18215
Input representation: 3 x 128 x 128
Latent representation: 4 x 16 x 16
Reconstruction MSE: 0.00074192

## Forward Diffusion
Diffusion timesteps: 1000
Beta schedule: linear
Beta start: 0.0001
Beta end: 0.02
Visualized timesteps: 0, 200, 500, 800, 1000

## Conditional Latent U Net
Base channels: 64
Channel multipliers: 1, 2, 4
Trainable parameters: 6486852
Time conditioning: sinusoidal embedding plus MLP
Normalization: GroupNorm
Activation: SiLU
Attention: latent 16 x 16 resolution and bottleneck
Class embeddings: 10 FashionMNIST classes plus 1 null class
Classifier free label drop probability: 0.1

## Training
Epochs: 50
Learning rate: 2e-05
Training samples: 60000
Initial noise prediction MSE: 0.40818276
Final noise prediction MSE: 0.09997120
Best epoch: 50
Best noise prediction MSE: 0.09997120
Training time seconds: 1738.195
Training time minutes: 28.970

## Sampling
Classifier free guidance scale: 7.5
Generated conditional samples: 64
Labels: FashionMNIST classes 0 through 9 repeated
Sampling diffusion steps: 1000

## Final Result
The VAE reconstruction MSE was 0.00074192. The latent denoiser completed all 50 required epochs. Noise prediction MSE decreased from 0.408183 to 0.099971. The final classifier free guidance sampler generated the required 64 image conditional FashionMNIST grid.
