# HW3 Metrics

SID4: 2046  
SEED: 2046  
SLICE: 46  
HP_ID: 0  
CLS_A: 6  
CLS_B: 3

## Dataset

Total tokens: 46  
Unique tokens: 40  
Next token training pairs: 45

## Unmasked Self Attention

Initial loss: 3.701041  
Final loss: 0.228414  
Minimum loss: 0.227368  
Total attention assigned to future positions: 13.463081

## Causal Masked Self Attention

Initial loss: 3.714020  
Final loss: 0.217287  
Minimum loss: 0.190088  
Maximum attention assigned to a future position: 0.0  
Total attention assigned to future positions: 0.0

## Attention Validation

Maximum row sum error for unmasked attention: approximately 2.38e-7  
Maximum row sum error for masked attention: approximately 2.38e-7

## Prompt Engineering

Zero Shot average output length: 98.0 words  
Few Shot average output length: 23.5 words  
Chain of Thought average output length: 130.0 words  
Zero Shot CoT average output length: 187.0 words  
Meta Prompting average output length: 413.5 words  
Tree of Thoughts average output length: 325.0 words
