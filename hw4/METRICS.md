# HW4 Metrics

## Dataset

| Metric | Value |
| --- | ---: |
| Tokenization | Character level |
| Corpus characters | 1,115,394 |
| Vocabulary size | 65 |
| Sequence length | 128 |
| Sliding windows | 1,115,266 |

## Model

| Metric | Value |
| --- | ---: |
| Hidden dimension | 128 |
| Attention heads | 4 |
| Head dimension | 32 |
| Decoder blocks | 2 |
| Feedforward dimension | 512 |
| Total parameters | 835,777 |
| Causal future attention weight | 0.0 |

## Training

| Setting | Value |
| --- | ---: |
| Optimizer | Adam |
| Learning rate | 0.0003 |
| Epochs | 5 |
| Batch size in recorded run | 32 |
| Samples per epoch | 8,000 |
| Batches per epoch | 250 |

### Training loss

| Epoch | Loss |
| ---: | ---: |
| 1 | 2.6183 |
| 2 | 2.3091 |
| 3 | 2.2014 |
| 4 | 2.1374 |
| 5 | 2.0882 |

Loss reduction: `0.5301`

## Decoding comparison

All samples use the prompt `ROMEO:`.

| Method | Unique bigram ratio | Observation |
| --- | ---: | --- |
| Greedy | 0.138 | Most deterministic, but repetitive |
| Temperature 0.5 | 0.145 | Very close to greedy |
| Temperature 1.0 | 0.412 | More varied while still readable |
| Temperature 1.5 | 0.681 | Highest diversity, more malformed words |
| Top k, k = 5 | 0.141 | Conservative and repetitive |
| Top k, k = 20 | 0.387 | More varied while staying controlled |

## Main findings

Greedy decoding produced the most consistent local text but repeated common phrases. Temperature 1.5 produced the most diverse output. Temperature 1.0 and top k with `k = 20` gave a better balance between variation and readability for this small character level model.
