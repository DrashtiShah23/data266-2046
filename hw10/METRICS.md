# HW10 Metrics

## Dataset

Dataset: ccdv/arxiv-summarization

Split: train

Documents used: first 100

Full document field: article

Reference summary field: abstract

## Chunking

Strategy 1: fixed word chunks

Chunk size: 200 words

Overlap: 40 words

Number of chunks: 4094

Strategy 2: sentence aware chunks

Target size: 200 words

Sentence overlap: 2 sentences

Number of chunks: 4993

## Embeddings and Vector Database

Embedding model: sentence-transformers/all-MiniLM-L6-v2

Embedding dimension: 384

Vector database: ChromaDB

Retrieval K: 5

## Baseline Retrieval

Fixed word Precision@5: 0.920

Fixed word Recall@5: 1.000

Fixed word MRR: 1.000

Sentence aware Precision@5: 1.000

Sentence aware Recall@5: 1.000

Sentence aware MRR: 1.000

## Cross Encoder Reranking

Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2

Candidate retrieval K: 15

Final K: 5

Fixed word reranked Precision@5: 1.000

Fixed word reranked Recall@5: 1.000

Fixed word reranked MRR: 1.000

Sentence aware reranked Precision@5: 1.000

Sentence aware reranked Recall@5: 1.000

Sentence aware reranked MRR: 1.000

## Conversational Memory

Follow up query: What kernel does it use?

Without memory Precision@5: 0.600

Without memory Recall@5: 1.000

Without memory MRR: 1.000

With memory Precision@5: 1.000

With memory Recall@5: 1.000

With memory MRR: 1.000

## Evaluation

Evaluation queries: 5

Relevance definition: a retrieved chunk is relevant when its source document ID matches the manually assigned relevant document for the evaluation query.

Main findings:

Sentence aware chunking improved baseline Precision@5 from 0.92 to 1.00.

Cross encoder reranking improved fixed word Precision@5 from 0.92 to 1.00.

Conversational memory improved dependent follow up Precision@5 from 0.60 to 1.00.
