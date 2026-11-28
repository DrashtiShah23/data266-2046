# HW12 Metrics

## Personal Parameters

SID4: 2046
SEED: 2046
SLICE: 46
HP_ID: 0
CLS_A: 6
CLS_B: 3

Hardware: Tesla T4

## Part 1: GPTQ

Base model: Qwen/Qwen3-0.6B

Revision: c1899de289a04d12100db370d81485cdf75e47ca

Evaluation dataset: WikiText2

| Configuration | Perplexity | p50 latency ms | p95 latency ms | Peak memory GB | Size on disk MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base LLM | 20.0273 | 1323.85 | 1732.72 | 1.123 | 1147.81 |
| GPTQ 4 bit LLM | 24.0744 | 3234.89 | 6273.59 | 0.528 | 527.17 |

Memory reduction: 52.98%

Disk size reduction: 54.07%

Perplexity increase: 20.21%

p50 latency increase: 144.35%

p95 latency increase: 262.07%

## Part 2: Knowledge Distillation

Teacher: instructor provided BERT model trained on AG News

Student: distilbert base uncased

Best temperature: 10

Best alpha: 0.25

### Temperature Sweep

| Temperature | Accuracy |
| ---: | ---: |
| 1 | 0.941842 |
| 2 | 0.941579 |
| 5 | 0.941447 |
| 10 | 0.942105 |

### Loss Weighting Sweep

| Alpha | Accuracy |
| ---: | ---: |
| 0.25 | 0.944342 |
| 0.50 | 0.942105 |
| 0.75 | 0.939474 |

### Teacher versus Best Student

| Configuration | Accuracy | p50 latency ms | p95 latency ms | Peak memory GB | Size on disk MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Teacher BERT | 0.944605 | 12.63 | 26.17 | 0.414 | 418.58 |
| Best distilled student | 0.944342 | 5.92 | 7.26 | 0.256 | 256.11 |

Accuracy drop: 0.0263 percentage points

p50 latency reduction: 53.13%

p95 latency reduction: 72.26%

Peak memory reduction: 38.16%

Disk size reduction: 38.81%

### Per Class Accuracy

| Class | Teacher accuracy | Student accuracy | Degradation |
| --- | ---: | ---: | ---: |
| World | 0.949474 | 0.964737 | -0.015263 |
| Sports | 0.988421 | 0.988947 | -0.000526 |
| Business | 0.913684 | 0.907368 | 0.006316 |
| Sci Tech | 0.926842 | 0.916316 | 0.010526 |

Largest degradation class: Sci Tech
