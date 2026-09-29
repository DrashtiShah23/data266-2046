# AI USE

## 1. Which parts did you use an assistant for, and which did you write yourself?

I used an AI assistant to help structure the DialogSum preprocessing, PEFT LoRA setup, training loop, debugging checks, and experiment code. I ran the notebook and experiments myself and checked the actual losses, gradients, generated summaries, and comparison tables.

## 2. Give one specific thing it produced that was wrong

The first training implementation used Seq2SeqTrainer. It appeared to run, but it reported a training loss of 0.000000 and the validation loss became NaN. Step level diagnostics also showed NaN gradient norms, so those results were invalid.

Wrong output from that implementation:

training loss: 0.000000
validation loss: NaN
grad_norm: NaN

## 3. How did you find out?

I noticed that a real summarization fine tuning run should not immediately report exactly zero loss while also producing poor or empty generation. I added sanity checks before training and then tested a manual forward and backward pass. The manual diagnostic produced a finite forward loss of about 2.46, 96 trainable tensors with finite gradients, and zero tensors with nonfinite gradients. That showed that the dataset, labels, model, LoRA adapters, and backward pass were working, while the Trainer path was the failing part.

## 4. What did you change, and why does your version work?

I replaced the failing Trainer path with an explicit PyTorch training loop. The loop performs the forward pass, checks that loss is finite, calls backward, clips and verifies the gradient norm, performs the AdamW optimizer step, advances the learning rate scheduler, and evaluates validation loss after each epoch. The corrected rank 4 run finished with training loss 1.624805 and validation loss 1.439975, and rank 16 also trained with finite losses.

Review this file before submission and adjust the wording so it reflects your own explanation of what you did and verified.
