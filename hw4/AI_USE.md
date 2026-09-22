# AI Use Statement

I used ChatGPT as a support tool while working on this assignment, mainly for clarification, reference finding, debugging, and review. I did not use a pretrained Transformer model or Hugging Face implementation in the submitted code.

## Where I used AI

### 1. Understanding the assignment constraints
I used ChatGPT to break the assignment into the required pieces and double check what counted as a manual implementation. In particular, I used it to verify that my solution should build the query, key, and value projections, split the hidden dimension into multiple heads, apply the causal mask, and recombine the heads without using `nn.MultiheadAttention` or `nn.Transformer`.

### 2. Finding and checking technical references
I used ChatGPT to point me toward references that were useful for checking the implementation details. The main references I consulted were:

* PyTorch `nn.Linear` documentation: https://docs.pytorch.org/docs/main/generated/torch.nn.Linear.html
* PyTorch `nn.LayerNorm` documentation: https://docs.pytorch.org/docs/main/generated/torch.nn.LayerNorm.html
* PyTorch `nn.GELU` documentation: https://docs.pytorch.org/docs/main/generated/torch.nn.modules.activation.GELU.html
* Andrej Karpathy's GPT implementation material for the general decoder block and causal self attention structure: https://github.com/karpathy/build-nanogpt

I used these as references for tensor shapes, layer behavior, residual connections, and the general structure of a decoder only Transformer. I still implemented the attention computation and causal masking explicitly in my own notebook to match the assignment rules.

### 3. Debugging the attention implementation
I used ChatGPT to reason through the expected tensor shapes for multi head attention. This was useful when checking the reshape and transpose steps from `(batch, sequence, hidden)` to `(batch, heads, sequence, head_dim)` and back again.

I also used it to check the causal masking logic. I added a verification step in the notebook that measures the total attention weight assigned to future positions. The result was `0.0`, which confirmed that the mask was working correctly.

### 4. Colab and dataset loading
I used ChatGPT to help make the dataset loading cell work cleanly in Google Colab. The final version checks for the professor provided `Shakespeare.txt` file and raises an error if it is missing instead of silently using a different dataset.

### 5. Training and decoding review
I used ChatGPT to review whether the training setup satisfied the rubric, including the use of Adam, five epochs, a sequence length of 128, and at least four attention heads. I also used it to check the logic for greedy decoding, temperature sampling, and top k sampling.

### 6. Interpreting the outputs and organizing the report
After running the notebook, I used ChatGPT to help organize the results into a short findings report. The actual loss values and generated text in the report come from the notebook run. I reviewed the outputs myself and used the generated samples to compare repetition, coherence, and diversity across the decoding methods.

## What I verified myself

I ran the notebook and checked the following directly:

* character level tokenization and vocabulary creation
* sliding window input and target pairs with sequence length 128
* four head masked self attention
* causal mask verification
* model parameter count and architecture settings
* five epoch training loss
* training loss curve
* greedy generation from `ROMEO:`
* three temperature settings
* two top k settings
* generated text and diversity measurements

AI was used as a development and review aid. The model training, numerical results, and generated outputs were produced by the PyTorch notebook.
