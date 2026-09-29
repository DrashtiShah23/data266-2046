# METRICS

## Personal Parameters

SID4: 2046
SEED: 2046
SLICE: 46
HP_ID: 0
CLS_A: 6
CLS_B: 3

## Setup

Base model: google/flan-t5-small
Dataset: DialogSum
Training subset: 1000 examples
Validation subset: 100 examples
Epochs: 2
Batch size: 4
Learning rate: 0.0002
LoRA alpha: 16
LoRA dropout: 0.05
Target modules: q and v

## LoRA Rank Results

| Rank | Trainable Parameters | Final Training Loss | Final Validation Loss | Training Time Seconds |
| --- | ---: | ---: | ---: | ---: |
| 4 | 172032 | 1.624805 | 1.439975 | 80.71 |
| 16 | 688128 | 1.612953 | 1.441840 | 68.46 |

Rank 16 reached a slightly lower training loss, while rank 4 reached a slightly lower validation loss with one quarter as many trainable parameters. The two fixed inference examples produced the same outputs for both ranks.

## Decoding Strategy Results

| Strategy | ROUGE1 | ROUGE2 | ROUGE L |
| --- | ---: | ---: | ---: |
| Greedy | 0.2515 | 0.0273 | 0.1866 |
| Beam search | 0.2817 | 0.0839 | 0.2132 |
| Sampling | 0.3053 | 0.0927 | 0.2285 |

Sampling used temperature 0.7 and top p 0.9.

## Temperature Results

| Temperature | ROUGE1 | ROUGE2 | ROUGE L |
| --- | ---: | ---: | ---: |
| 0.3 | 0.2637 | 0.0423 | 0.2181 |
| 0.5 | 0.2853 | 0.0718 | 0.2057 |
| 0.7 | 0.3053 | 0.0927 | 0.2285 |
| 1.0 | 0.2690 | 0.0680 | 0.2095 |

Temperature 0.7 had the highest measured ROUGE L on the ten fixed examples.
