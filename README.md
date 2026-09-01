# DATA 266 — Homework 1

**Student:** Drashti Shah · **Full SID:** 013852046 · **SID4:** 2046
**SEED:** 2046 · **SLICE:** 46 · **HP_ID:** 0 · **CLS_A:** 6 · **CLS_B:** 3
**Training seeds:** 2046, 2047, 2048

Autoregressive models (written), a PyTorch/TensorFlow diabetes classifier (baseline vs.
HP_ID 0 capacity-reduced model), and a CUDA tiled matrix-multiplication kernel
benchmarked against an OpenMP CPU reference.

## Results at a glance

| | Baseline | Modified (HP_ID 0: [32] hidden) |
|---|---|---|
| PyTorch mean test accuracy | 0.7456 ± 0.0152 | 0.7485 ± 0.0051 |
| TensorFlow mean test accuracy | 0.7310 ± 0.0051 | 0.7515 ± 0.0134 |

| Matrix size | CPU (ms) | GPU end-to-end (ms) | Speedup |
|---|---|---|---|
| 256 | 11.18 | 0.77 | 14.5× |
| 1024 | 305.81 | 7.92 | 38.6× |
| 4096 | 22593.20 | 290.28 | 77.8× |

Full tables, loss diagnostics, and Nsight Compute analysis: [`METRICS.md`](METRICS.md).
Full write-up: [`Homework1_Report.docx`](Homework1_Report.docx).

## Repository contents

**Neural network**
- [`neural_networks.ipynb`](neural_networks.ipynb) — PyTorch + TensorFlow experiments, executed with outputs intact
- [`neural_metrics.csv`](neural_metrics.csv) — all 12 individual run accuracies
- [`loss_histories.json`](loss_histories.json) — per-epoch train/val loss, seed 2046
- [`figures/correlation_matrix.png`](figures/correlation_matrix.png), [`figures/feature_distributions.png`](figures/feature_distributions.png)
- [`figures/pytorch_loss_curves.png`](figures/pytorch_loss_curves.png), [`figures/tensorflow_loss_curves.png`](figures/tensorflow_loss_curves.png)

**CUDA**
- [`cuda.ipynb`](cuda.ipynb) — build, run, and profiling driver
- [`matrix_mul.cu`](matrix_mul.cu) — tiled shared-memory kernel + OpenMP CPU reference
- [`cuda_metrics.csv`](cuda_metrics.csv), [`cuda_run_output.txt`](cuda_run_output.txt) — timing results
- [`profiler_output.txt`](profiler_output.txt), [`ncu_details.txt`](ncu_details.txt), [`hw1_ncu.ncu-rep`](hw1_ncu.ncu-rep) — Nsight Compute output (basic + full)

**Data, logs, and writeups**
- [`diabetes.csv`](diabetes.csv) — 759 rows, 8 features, binary `Outcome`
- [`RUN_LOG.txt`](RUN_LOG.txt) — console output from the runs that produced the reported numbers
- [`METRICS.md`](METRICS.md) — all required measurement tables
- [`AI_USE.md`](AI_USE.md) — AI-use disclosure (Section 0.5)
- [`Homework1_Report.docx`](Homework1_Report.docx) — final report

## Reproducing

Both notebooks run top to bottom in Google Colab with no manual intervention:
`neural_networks.ipynb` needs `diabetes.csv` in the same directory; `cuda.ipynb` needs a
T4 GPU runtime and `matrix_mul.cu` in the same directory (`nvcc -O3 -std=c++17
-Xcompiler -fopenmp -lineinfo matrix_mul.cu -o matrix_mul`).