# METRICS

## Personal Parameters

| Parameter | Value |
| --- | ---: |
| SID4 | 2046 |
| SEED | 2046 |
| SLICE | 46 |
| HP_ID | 0 |
| CLS_A | 6 (Shirt) |
| CLS_B | 3 (Dress) |

## Experimental Configuration

| Setting | FC Autoencoder | VAE |
| --- | ---: | ---: |
| Epochs | 15 | 15 |
| Latent dimension | 32 | 32 |
| Batch size | 256 | 256 |
| Optimizer | Adam | Adam |
| Learning rate | 0.001 | 0.001 |
| Training objective | MSE | BCE + KL divergence |

## Common Test Reconstruction Metrics

Both models were evaluated on the complete 10,000 image FashionMNIST test set using the same per image MSE calculation.

| Model | Test MSE | Test RMSE |
| --- | ---: | ---: |
| FC Autoencoder | 0.012584 | 0.112177 |
| VAE | 0.016428 | 0.128171 |

## Focus Class Results

| Class | FC AE Mean MSE | VAE Mean MSE | FC AE Median MSE | VAE Median MSE |
| --- | ---: | ---: | ---: | ---: |
| Shirt (6) | 0.012350 | 0.015830 | 0.009588 | 0.012589 |
| Dress (3) | 0.011551 | 0.015666 | 0.009450 | 0.013641 |

## VAE Test Objective

The VAE training objective is reported separately because BCE plus KL divergence is not directly comparable with the FC Autoencoder MSE objective.

Test total VAE loss per sample: 244.774

Test reconstruction BCE per sample: 232.075

Test KL divergence per sample: 12.699

The final model comparison therefore uses the common reconstruction MSE reported above.
