# AI USE

## 1. Which parts did you use an assistant for, and which did you write yourself?

I used an AI assistant mainly to help organize the notebook, structure the PyTorch implementation for the FC Autoencoder and VAE, and review how the two models should be evaluated fairly. It also helped me organize the plots and failure analysis and explain the difference between reconstruction accuracy and the VAE's generative objective.

I ran the notebook myself in Colab and checked the outputs at each stage. I verified my assigned parameters, dataset sizes, tensor shapes, training behavior, reconstruction outputs, test metrics, and the final comparison between the two models. I also changed the notebook organization and visualizations while working through the assignment rather than using the assistant output without checking it.

## 2. Give one specific thing it produced that was wrong.

I did not encounter a model implementation produced by the assistant that gave an incorrect tensor shape, invalid loss, or runtime error in the final experiment.

However, one issue I specifically checked was the proposed comparison between the FC Autoencoder and VAE. Their raw training losses cannot be compared directly because the FC Autoencoder was trained using MSE, while the VAE was trained using reconstruction BCE plus KL divergence. Treating those raw losses as if they measured the same quantity would have produced a misleading model comparison.

## 3. How did you find out? What did the failure look like?

I verified the loss definitions and inspected the actual outputs. The FC Autoencoder common test MSE was 0.012584, while the VAE test objective was 244.774 per sample and contained both reconstruction BCE and KL divergence. The difference in definition and scale showed that these values could not be used directly to decide which model reconstructed images better.

I therefore evaluated both trained models over the complete 10,000 image FashionMNIST test set using the same per image MSE calculation. Under this common metric, the FC Autoencoder obtained an MSE of 0.012584 and the VAE obtained 0.016428.

## 4. What did you change, and why does your version work?

For the final model comparison, I used the same reconstruction MSE calculation for both models instead of comparing their original training objectives. For the VAE, I used the encoder mean as the latent representation during this evaluation rather than randomly sampling a latent vector. This makes the reconstruction comparison deterministic.

This version is appropriate because both models are evaluated on the same test images with the same reconstruction metric. The VAE's BCE plus KL objective is still reported separately because it is relevant to VAE training, but it is not treated as directly comparable to the FC Autoencoder's MSE.

## Verification performed

Personal parameters were recomputed and verified from SID4 = 2046.

FashionMNIST dataset sizes were verified.

The FC Autoencoder reconstruction shape was verified against its input shape.

The VAE latent mean and log variance shapes were verified.

The VAE reconstruction shape was verified.

Training and validation behavior was inspected across all 15 FC AE epochs and 15 VAE epochs.

Both models were evaluated over the complete FashionMNIST test set.

The same per image MSE calculation was used for the final model comparison.

The VAE latent mean was used for deterministic reconstruction evaluation.

The assigned Shirt and Dress classes were evaluated separately.

The highest error reconstructions were inspected for both models.
