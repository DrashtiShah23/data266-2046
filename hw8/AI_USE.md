# AI Use Disclosure

AI assistance was used during the development of Homework 8.

## Tool Used

ChatGPT

## How AI Was Used

AI was used to help interpret the assignment requirements and organize the implementation into incremental development steps.

AI assisted with the implementation and explanation of the forward diffusion process, timestep conditioned CNN architecture, DDPM training procedure, DDPM reverse sampling, and DDIM sampling.

AI was also used for debugging and experimental analysis. After the first trained model achieved a low noise prediction loss but generated poorly structured samples, AI assisted with identifying the limitation, designing a revised encoder and decoder style CNN with a larger effective receptive field, and planning a second controlled training experiment.

AI assisted with interpreting training metrics, sampling runtime measurements, and qualitative differences between DDPM and DDIM outputs.

AI also helped organize the Git workflow, experiment log, metrics documentation, and checkpoint management.

## Verification and Execution

All notebook cells were executed in Google Colab.

Training results, generated images, runtime measurements, and reported metrics were obtained from the executed notebook rather than generated or estimated by AI.

The implementation was tested incrementally, and model outputs were visually inspected before proceeding to later stages.

## Development Process

The work was completed incrementally with separate Git commits for workspace setup, forward diffusion, model implementation, training, experiment records, checkpoints, sampling, and final documentation.

The initial model and its unsuccessful sampling results were retained as part of the experimental record rather than being removed after the architecture was revised.
