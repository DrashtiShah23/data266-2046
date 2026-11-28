# AI Use

I used ChatGPT to help structure the experiment workflow, implement the GPTQ evaluation and benchmarking code, implement the knowledge distillation loss and training loop, and organize the required measurements and analysis. I ran the experiments myself in Google Colab on a Tesla T4, inspected the outputs, and used the measured results from my executed notebook.

One specific issue in the generated workflow was the GPTQ environment setup. An early dependency installation resulted in a TorchAudio and PyTorch CUDA version mismatch, which caused GPTQModel imports to fail. The error reported that PyTorch and TorchAudio had been compiled with different CUDA versions. I identified the problem from the runtime stack trace and restarted with a cleaner dependency setup rather than using results from the inconsistent environment.

Another issue occurred when peak GPU memory for the teacher and student initially appeared identical because both models were simultaneously resident on the GPU. I recognized that this did not represent individual model memory. I unloaded both models and measured each model separately relative to the GPU allocation baseline. This produced individual peak measurements of 0.414 GB for the teacher and 0.256 GB for the student.

I verified that the base and GPTQ models used the same model ID, revision, WikiText2 evaluation, prompts, and Tesla T4 hardware. For distillation, I verified that each sweep condition initialized a fresh DistilBERT student using SEED 2046 while keeping the other training settings fixed. I also verified that the final teacher and student measurements used the same AG News test set and hardware.
