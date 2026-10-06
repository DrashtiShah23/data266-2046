# METRICS

## Personal Parameters

SID4: 2046  
SEED: 2046  
SLICE: 46  
HP_ID: 0  
CLS_A: 6  
CLS_B: 3  

## Common Setup

Dataset: STL10  
Backbone for all three approaches: ResNet18  
Pretrained weights: None  
Labeled subset for downstream classification: 500 images  

## Part A: Supervised Limited Labels

| Metric | Value |
| --- | ---: |
| Labeled training images | 500 |
| Epochs | 12 |
| Final training loss | 0.7544 |
| Final training accuracy | 75.40% |
| Test loss | 1.5480 |
| Test accuracy | 44.40% |
| Training time | 11.11 s |

## Part B: Rotation SSL

| Metric | Value |
| --- | ---: |
| Unlabeled pretraining images | 100000 |
| Rotation classes | 4 |
| Rotation pretraining epochs | 15 |
| Final rotation loss | 0.000585 |
| Final rotation accuracy | 100.00% |
| Rotation pretraining time | 1218.26 s |
| Linear evaluation labeled images | 500 |
| Linear evaluation epochs | 20 |
| Trainable encoder parameters during linear evaluation | 0 |
| Trainable linear classifier parameters | 5130 |
| Test loss | 2.0612 |
| Test accuracy | 28.86% |

## Part C: SimCLR

Augmentations: RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale.

| Metric | Value |
| --- | ---: |
| Unlabeled pretraining images | 20000 |
| Temperature tau | 0.2 |
| Projection dimension | 128 |
| Contrastive pretraining epochs | 15 |
| Final contrastive loss | 2.202351 |
| Contrastive pretraining time | 1270.80 s |
| Linear evaluation labeled images | 500 |
| Linear evaluation epochs | 20 |
| Trainable encoder parameters during linear evaluation | 0 |
| Trainable linear classifier parameters | 5130 |
| Test loss | 1.4622 |
| Test accuracy | 47.43% |

## Part D: Nearest Neighbor Retrieval

Query indices: 3383, 2661, 2013  
Query classes: cat, dog, horse  
Similarity: cosine similarity  
Neighbors per query: 5  

| Encoder | Test Accuracy | Top 5 Same Class Neighbor Match |
| --- | ---: | ---: |
| Supervised | 44.40% | 40.00% |
| Rotation SSL | 28.86% | 26.67% |
| SimCLR | 47.43% | 6.67% |

### Query level same class matches

| Query | Supervised | Rotation SSL | SimCLR |
| --- | ---: | ---: | ---: |
| Cat | 1/5 | 1/5 | 0/5 |
| Dog | 1/5 | 0/5 | 0/5 |
| Horse | 4/5 | 3/5 | 1/5 |

SimCLR produced the highest downstream linear evaluation accuracy. For the three selected nearest neighbor queries, the supervised encoder produced the strongest same class local neighborhoods.
