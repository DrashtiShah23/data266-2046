# HW11 Metrics

## Dataset

Dataset: CIFAR 10

Similarity experiment images: 1000

Images per class: 100

Zero shot classification test images: 10000

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## CLIP

Model: openai/clip-vit-base-patch32

Device: cuda

Image text embedding dimension: 512

Embeddings: L2 normalized

Similarity metric: cosine similarity

## Caption Similarity

Caption: a photo of an airplane flying in the sky

Highest mean similarity class: airplane

Highest class mean similarity: 0.248972

Airplane mean similarity: 0.248972

## Zero Shot Classification

Prompt template: a photo of a {class}

Overall accuracy: 0.8880

Overall accuracy percent: 88.80%

Best class: automobile

Best class accuracy: 0.9760

Worst class: deer

Worst class accuracy: 0.7650

Classification runtime seconds: 49.36

## Per Class Accuracy

airplane: 0.9010
automobile: 0.9760
bird: 0.8880
cat: 0.8600
deer: 0.7650
dog: 0.8840
frog: 0.7740
horse: 0.9760
ship: 0.9550
truck: 0.9010
