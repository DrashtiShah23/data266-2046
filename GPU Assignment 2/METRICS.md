# GPU Assignment II Metrics

## Student Specific Constants

SID4: 2046

SEED: 2046

SLICE: 46

HP_ID: 0

CLS_A: 6

CLS_B: 3

## Hardware

GPU: NVIDIA GeForce RTX 5090

GPU memory: approximately 31.84 GiB

PyTorch used for original HW4.a BERT baseline: 2.7.1+cu128

vLLM environment after compatibility update: PyTorch 2.14.0+cu130

vLLM version: 0.33.0

## Dataset

Dataset: cardiffnlp/tweet_eval

Configuration: sentiment

Source row range: 46 through 5045

Total selected rows: 5000

Training rows: 3500

Validation rows: 750

Test rows: 750

Split seed: 2046

The identical dataset range, split indices, and seed were reused across HW4.a and HW4.5.

## HW4.a BERT Baseline

Model: bert-base-uncased

Baseline batch size: 16

Epoch budget: 4

Best validation epoch: 3

Best validation accuracy: 0.6666666666666666

Peak GPU memory GB: 2.477545738220215

Training wall clock seconds: 49.511319

Training GPU hours: 0.01375314

## HW4.5 Part A

Largest successfully measured physical batch: 256

Batch 512 did not complete within the practical measurement window and was interrupted.

Throughput at batch 256: 681.182 samples per second

Peak memory at batch 256: 20.760 GB

Time per epoch at batch 256: 5.138 seconds

First measured batch with less than 5 percent throughput improvement: 256

## HW4.5 Part B

### Mixed Precision

Throughput: 1465.051 samples per second

Peak memory: 15.189 GB

Memory reduction versus baseline: 26.83 percent

Throughput change versus baseline: 115.07 percent

### Best Combination

Configuration: mixed precision plus gradient checkpointing

Physical batch: 256

Throughput: 1226.808 samples per second

Peak memory: 5.106 GB

Memory reduction versus baseline: 75.40 percent

Throughput change versus baseline: 80.10 percent

The best combination reduced measured peak memory substantially while also improving throughput relative to the unoptimized batch 256 baseline.

The 8 bit optimizer did not improve memory efficiency in this experiment.

## HW4.5 Part C

Part C skipped (optional).

## HW4.5 Part D

GPU: NVIDIA GeForce RTX 5090

The vLLM serving benchmark was measured at batch sizes 1, 8, and 32.

Maximum successfully tested input context at batch size 8: 1984 tokens.

The vLLM engine was configured with max_model_len equal to 2048 and generation length equal to 32 tokens.

The context experiment therefore reports the maximum successfully tested context under the configured model limit rather than claiming the physical maximum context supported by the RTX 5090.

vLLM preallocated its KV cache during engine initialization. As a result, request level before and after GPU memory measurements remained approximately constant because longer requests consumed capacity inside the already reserved cache pool.

## HW4.5 Part E

Measured baseline fine tune GPU hours: 0.01375314

Recommended configuration: mixed precision plus gradient checkpointing.

Reason: it provided the strongest measured memory efficiency tradeoff, reducing peak memory by approximately 75.40 percent while increasing throughput by approximately 80.10 percent relative to the baseline batch 256 configuration.

Mixed precision alone produced the highest measured throughput, while the combined configuration provided substantially lower memory usage.

## Notes

All reported benchmark values were obtained from executed experiments on the NVIDIA GeForce RTX 5090.

HW4.5 values are HW4.5 measurements only and are not labeled as an HW5 baseline.
