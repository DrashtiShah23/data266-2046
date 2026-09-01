# Homework 1 Metrics — data266-2046

## Personal parameters
- Student: Drashti Shah | Full Student ID: 013852046
- SID4 = 2046, SEED = 2046, SLICE = 46, HP_ID = 0, CLS_A = 6, CLS_B = 3
- Training seeds = [2046, 2047, 2048]

## Dataset and split
- diabetes.csv, headerless, shape (759, 9), 0 missing values, 0 duplicate rows
- Class distribution: Outcome 0 → 263, Outcome 1 → 496

| Subset | Rows | Percent | Positive rate |
|---|---|---|---|
| Training | 531 | 70.0% | 0.653 |
| Validation | 114 | 15.0% | 0.658 |
| Testing | 114 | 15.0% | 0.649 |

Split: random_state=SEED (2046), stratified, reused for every comparison.
Scaler: StandardScaler fit on training data only.

## Correlation with Outcome (real, from df.corr())
Glucose -0.49, BMI -0.31, Pregnancies -0.22, SkinThickness -0.19,
DiabetesPedigreeFunction -0.16, Insulin -0.16, BloodPressure -0.15, Age -0.11.
Strongest predictor-predictor pair: SkinThickness–Insulin at 0.47.
Note: every feature correlates negatively with Outcome, including Glucose — Outcome in
this file does not follow the conventional "1 = positive/at-risk" clinical labeling.

## Model configurations
| Model | Hidden layers | LR | Epochs | Batch | Output/loss |
|---|---|---|---|---|---|
| Baseline | [64, 32] | 0.001 | 30 | 32 | 1 raw logit, BCE-with-logits |
| Modified (HP_ID 0) | [32] | 0.001 | 30 | 32 | 1 raw logit, BCE-with-logits |

## Individual neural-network runs
| Framework | Model | Seed | Test accuracy |
|---|---|---|---|
| PyTorch | Baseline | 2046 | 0.7544 |
| PyTorch | Baseline | 2047 | 0.7281 |
| PyTorch | Baseline | 2048 | 0.7544 |
| PyTorch | Modified (HP_ID 0) | 2046 | 0.7456 |
| PyTorch | Modified (HP_ID 0) | 2047 | 0.7456 |
| PyTorch | Modified (HP_ID 0) | 2048 | 0.7544 |
| TensorFlow | Baseline | 2046 | 0.7281 |
| TensorFlow | Baseline | 2047 | 0.7368 |
| TensorFlow | Baseline | 2048 | 0.7281 |
| TensorFlow | Modified (HP_ID 0) | 2046 | 0.7632 |
| TensorFlow | Modified (HP_ID 0) | 2047 | 0.7368 |
| TensorFlow | Modified (HP_ID 0) | 2048 | 0.7544 |

## Three-seed mean / standard deviation
| Framework | Model | Mean | Std dev (n=3) |
|---|---|---|---|
| PyTorch | Baseline | 0.7456 | 0.0152 |
| PyTorch | Modified (HP_ID 0) | 0.7485 | 0.0051 |
| TensorFlow | Baseline | 0.7310 | 0.0051 |
| TensorFlow | Modified (HP_ID 0) | 0.7515 | 0.0134 |

## Seed-2046 loss diagnostics
| Framework | Model | Best val epoch | Min val loss | Final train loss | Final val loss |
|---|---|---|---|---|---|
| PyTorch | Baseline | 19 | 0.4778 | 0.3857 | 0.4829 |
| PyTorch | Modified (HP_ID 0) | 29 | 0.5000 | 0.4304 | 0.5015 |
| TensorFlow | Baseline | 28 | 0.4954 | 0.3652 | 0.4961 |
| TensorFlow | Modified (HP_ID 0) | 20 | 0.5003 | 0.4155 | 0.5017 |

## Environment note
PyTorch training ran on CPU despite a Tesla T4 being available in the Colab session,
because the training code never calls `.to('cuda')`. Disclosed in `AI_USE.md` as a real
assistant-produced error. Does not violate any assignment requirement — GPU is not
required for the neural-network part.

## CUDA configuration
GPU: Tesla T4. Kernel: 16×16 tiled, 256 threads/block, shared-memory reuse, boundary-checked.
CPU reference: OpenMP, i-k-j loop order (see AI_USE.md for why), mean of 3 runs.
GPU: 1 untimed warm-up + mean of 10 runs (kernel and transfer timed separately via CUDA events).

## CUDA timing table
| Matrix size | CPU (ms) | GPU kernel (ms) | H2D+D2H (ms) | GPU end-to-end (ms) | Speedup | Max abs error |
|---|---|---|---|---|---|---|
| 256 | 11.1777 | 0.1011 | 0.6681 | 0.7692 | 14.53 | 2.289e-05 |
| 1024 | 305.807 | 4.9215 | 2.9969 | 7.9184 | 38.62 | 9.155e-05 |
| 4096 | 22593.2 | 244.861 | 45.4202 | 290.281 | 77.83 | 3.662e-04 |

## Nsight Compute summary (1024×1024)
Grid (64,64,1) × Block (16,16,1) = 1,048,576 threads. Registers/thread: 40. Static shared
mem/block: 2.05 KB. Theoretical occupancy: 100%. Achieved occupancy: 98.67–98.69%.
Compute throughput: 74.53–74.58%. Memory throughput: 74.53–74.58%. DRAM throughput: ~10.8%.
Profiled duration 5.78 ms vs. unprofiled CUDA-event kernel time 4.92 ms — difference is
Nsight Compute's replay-pass instrumentation overhead (9 passes for --set basic, 31 for
--set full), not a discrepancy in the underlying kernel. Scheduler stats: only ~25.2% of
cycles have an eligible warp, with ~55% of stall cycles attributed to the shared-memory
(MIO) instruction queue being full — kernel is somewhat latency-bound on shared-memory
traffic despite near-100% occupancy.

## GPU crossover analysis
Smallest tested size with an end-to-end GPU benefit: 256 (speedup 14.53×). Not at size
zero because GPU execution has fixed launch/sync/transfer costs a near-zero workload
can't amortize.

## Pending
- pytorch_loss_curves.png / tensorflow_loss_curves.png: generated in Colab but not yet
  reviewed here — swap into the report once uploaded (numbers above are already final).
