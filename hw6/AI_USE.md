# AI USE

## 1. Which parts did you use an assistant for, and which did you write yourself?

I used an AI assistant to help organize the notebook, write parts of the PyTorch training and evaluation code, set up the Rotation SSL and SimCLR pipelines, and build the cosine nearest neighbor analysis. I ran the experiments myself in Colab, checked the losses and accuracies, verified that the encoders were frozen during linear evaluation, and reviewed the generated nearest neighbor visualizations and final comparisons.

## 2. Give one specific thing it produced that was wrong

One issue I had to catch was how to interpret the nearest neighbor result for SimCLR. SimCLR had the best downstream test accuracy, 47.43%, but for the three selected query images it had only a 6.67% same class top 5 neighbor match rate. It would have been incorrect to automatically describe the best classifier as also having the best nearest neighbor representation.

Observed output:

Supervised top 5 label match rate: 40.00%
Rotation SSL top 5 label match rate: 26.67%
SimCLR top 5 label match rate: 6.67%

At the same time, the test accuracies were:

Supervised: 44.40%
Rotation SSL: 28.86%
SimCLR: 47.43%

## 3. How did you find out?

I found it by checking the actual nearest neighbor outputs and labels instead of assuming that classification accuracy and nearest neighbor class matching would rank the encoders the same way. The three fixed queries were a cat, dog, and horse. SimCLR retrieved zero same class neighbors for the cat, zero for the dog, and one for the horse, even though its frozen representation gave the highest linear evaluation accuracy.

## 4. What did you change, and why does your version work?

I changed the final interpretation so that it treats linear evaluation and nearest neighbor retrieval as two different views of representation quality. Linear evaluation measures whether the overall embedding space can be separated effectively by a learned linear classifier. The nearest neighbor visualization measures the local cosine neighborhood around a few specific query images. I also explicitly note that only three queries were examined, so the 6.67% SimCLR neighbor match rate should not be generalized to the entire test set.

Before submission, I reviewed the code and outputs so I can explain the supervised training, rotation pretext task, frozen linear evaluation, SimCLR contrastive objective, and cosine nearest neighbor retrieval.
