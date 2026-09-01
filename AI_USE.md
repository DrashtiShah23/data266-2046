# AI Use Disclosure — Homework 1 (data266-2046)

## 1. Which parts did I use an assistant for, and which did I write/verify myself?

I used an assistant to draft the structure of both notebooks cell-by-cell: the PyTorch
and TensorFlow training loops, the `matrix_mul.cu` CUDA benchmark harness, and the
Colab driver code around it (compile, run, profile, log). I wrote/verified myself: the
personal-parameter derivation (SID4=2046 → SEED/SLICE/HP_ID/CLS_A/CLS_B) checked by hand
against Section 0.1's formulas before running anything; every numerical result in this
repo, by actually executing both notebooks in Colab and pasting the real output back
rather than accepting anything pre-written; and the three specific defects below, which
I either caught by reading the code against the assignment spec or by observing a real
symptom while running it.

## 2. Specific incorrect output #1 — CPU reference had a pathological memory-access pattern

**What happened:** The CPU reference implementation in `matrix_mul.cu` used an `i-j-k`
loop order:

```cpp
for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (int k = 0; k < n; ++k)
            acc += A[i*n+k] * B[k*n+j];   // B accessed column-wise in the inner loop
        C[i*n+j] = acc;
    }
```

**Wrong output:** not a crash or an exception — the CUDA benchmark cell just produced no
output at all for several minutes at N=4096, with the cell showing as still running.
Nothing printed, nothing errored; it looked identical to a hung/broken cell.

**How I found out:** I compared this against what N=256 and N=1024 took (well under a
second each), and against the fact that CPU work should scale roughly with N³ which was going
from 1024 to 4096 is a 64× increase in arithmetic, not the >100× slowdown-to-apparent-hang
I was seeing. That pointed at the memory-access pattern rather than raw arithmetic volume:
`B[k*n+j]` with `j` fixed in the innermost loop walks straight down a column of B, jumping
`n*4` bytes (16 KB at n=4096) between consecutive accesses, a near-guaranteed cache miss
on almost every read, a well-known worst case for naive triple-loop matrix multiply.

**What I changed and why it works:** Reordered to `i-k-j`, so both B and C are accessed
row-wise (stride-1) in the innermost loop:

```cpp
for (int i = 0; i < n; ++i)
    for (int k = 0; k < n; ++k) {
        float a_ik = A[i*n+k];
        for (int j = 0; j < n; ++j)
            C[i*n+j] += a_ik * B[k*n+j];   // row-wise access, cache-friendly
    }
```

This requires zero-initializing `C` first (values now accumulate across the `k` loop
instead of being computed in one pass per output element). Measured result: the full
3-repeat CPU benchmark at N=4096 went from an unmeasured multi-minute hang to 22.6
seconds total — I verified this by timing the actual run, not by assumption.

## 3. Specific incorrect output #2 — PyTorch silently trained on CPU instead of GPU

**What happened:** `train_torch()` builds tensors with `torch.from_numpy(...)` and the
model with no device argument, there's no `.to('cuda')` or `.cuda()` call anywhere in
the function.

**Wrong output:** Ran the neural-network notebook twice in two different Colab
sessions — once with a CPU-only runtime (`PyTorch CUDA available: False`) and once with
a T4 GPU attached (`PyTorch CUDA available: True`, `PyTorch device: Tesla T4`). All 12
PyTorch/TensorFlow test-accuracy values, and all four loss-diagnostic rows, came back
bit-for-bit identical between the two runs.

**How I found out:** Identical results between a CPU-only session and a GPU session
shouldn't happen if the GPU is actually doing the computation — CPU (MKL) and GPU
(cuBLAS/cuDNN) floating-point kernels don't reproduce bit-identical results even with the
same random seed, so after 30 epochs of training I'd expect at least a few predictions
near the 0.5 threshold to flip. Getting the exact same accuracy twice was the tell that
the "GPU available: True" session wasn't actually using the GPU for training.

**What I changed:** Nothing needed to be fixed for correctness — the assignment doesn't
require GPU for the neural-network part, and CPU training is valid. I documented the
cause (missing `.to(device)` calls) instead of leaving the identical-results coincidence
unexplained, since an unexplained "too clean" result is exactly the kind of thing that
should raise suspicion rather than be accepted at face value.

## 4. Specific incorrect output #3 — TensorFlow model didn't match the spec's loss/output requirement

**What happened:** An earlier draft of `build_tf_model()` used a `sigmoid` activation on
the output layer paired with `BinaryCrossentropy(from_logits=False)`.

**How I found out:** Not a runtime error, I caught this by re-reading the assignment
text, which explicitly requires "one raw output logit" and "binary cross-entropy with
logits" for both frameworks. A sigmoid-activated output with a non-logits loss is
mathematically similar but is not what was specified, and it made the TensorFlow model
inconsistent with the PyTorch side, which already used `BCEWithLogitsLoss` on a raw
logit.

**What I changed:** Removed the output activation (`Dense(1)` with no activation) and
switched to `BinaryCrossentropy(from_logits=True)`, with predictions computed by
applying `sigmoid` manually at inference time before thresholding at 0.5. This matches
the spec exactly and keeps both frameworks on the same loss formulation.

## What I verified myself, beyond fixing errors

- Checked the SID4 → SEED/SLICE/HP_ID/CLS_A/CLS_B arithmetic by hand.
- Confirmed the HP_ID 0 mapping (hidden layers [32], same LR, same epochs) matches
  Table 0.1 / the HW1-specific mapping exactly.
- Confirmed the 70/15/15 split uses `random_state=SEED` and that the same split and
  fitted scaler are reused for every framework/model comparison, not re-split per model.
- Re-ran the CUDA correctness check (max absolute error vs. the CPU reference) at all
  three sizes and confirmed it stays below 4×10⁻⁴, consistent with float32 rounding
  rather than a broken kernel.
- Confirmed real correlation values (Glucose -0.49, BMI -0.31, etc.) before writing any
  claim about which feature separates the classes best, rather than guessing from the
  figure alone.
