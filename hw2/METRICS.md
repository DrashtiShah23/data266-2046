# HW2 Metrics

## Personal parameters

SID4: 2046

SEED: 2046

SLICE: 46

HP_ID: 0

CLS_A: 6

CLS_B: 3

## Part 1: Embedding transfer learning

I compared the nearest neighbors from the original Google News embeddings with the neighbors after continuing training on the IMDB reviews.

| Word   | Rank | Original Neighbor | Original Similarity | Fine Tuned Neighbor | Fine Tuned Similarity |
| ====== | ==== | ================= | =================== | =================== | ===================== |
| cast   | 1    | casts             | 0.7219              | casted              | 0.5897                |
| cast   | 2    | casting           | 0.7188              | supporting          | 0.5843                |
| cast   | 3    | Cast              | 0.6638              | casting             | 0.5738                |
| score  | 1    | scoring           | 0.7197              | nicolai             | 0.6504                |
| score  | 2    | scores            | 0.6596              | music               | 0.6341                |
| score  | 3    | scored            | 0.6384              | ennio               | 0.6333                |
| plot   | 1    | plots             | 0.7625              | storyline           | 0.7203                |
| plot   | 2    | Plot              | 0.6524              | story               | 0.6824                |
| plot   | 3    | plotting          | 0.6328              | plots               | 0.6641                |
| screen | 1    | screens           | 0.7729              | screens             | 0.5620                |
| screen | 2    | onscreen          | 0.6115              | onscreen            | 0.5452                |
| screen | 3    | LCD_screen        | 0.5599              | stage               | 0.4612                |
| review | 1    | reviewed          | 0.6630              | comment             | 0.6832                |
| review | 2    | reviewing         | 0.6610              | maltin              | 0.6370                |
| review | 3    | reviews           | 0.6380              | reviews             | 0.6298                |

### Original vector compared with the fine tuned vector

| Word   | Original vs Fine Tuned Cosine Similarity | Shift Amount |
| ====== | ======================================== | ============ |
| cast   | 0.7127                                   | 0.2873       |
| score  | 0.5552                                   | 0.4448       |
| plot   | 0.6846                                   | 0.3154       |
| screen | 0.6634                                   | 0.3366       |
| review | 0.5425                                   | 0.4575       |

The word that shifted the most was **review** because it had the lowest cosine similarity between its original and fine tuned vectors.

The word that shifted the least was **cast** because it had the highest cosine similarity.

## Part 2: RAG retrieval results

| Question                                                                      | Correct Source     | Top 3 Contains Correct Information | First Relevant Rank |
| ============================================================================= | ================== | ================================== | =================== |
| Who directed the film Parasite?                                               | Parasite 2019 film | Yes                                | 1                   |
| In what year was Toy Story released?                                          | Toy Story          | Yes                                | 1                   |
| Who played Neo in The Matrix?                                                 | The Matrix         | Yes                                | 1                   |
| Who directed Pulp Fiction?                                                    | Pulp Fiction       | Yes                                | 2                   |
| Which director made the film about entering people's dreams called Inception? | Inception          | Yes                                | 1                   |

The correct answer passage appeared in the top three retrieved chunks for 5 of the 5 questions. My Retrieval Success Rate was **100.00%**.

The original setup used a chunk size of 500 with an overlap of 50. For two questions, I changed this to a chunk size of 800 with an overlap of 100 and compared the retrieved passages and answers.

### RAG failure analysis

One issue I noticed was that retrieving the correct information did not always mean it appeared first. For the Pulp Fiction question, the passage containing Quentin Tarantino was retrieved, but another result ranked above it.

I also saw cases where the retriever selected a passage from the correct movie but the passage itself did not contain the actual answer. Those results are related to the question, but they still add unnecessary context for the language model.

## Part 3: Training optimization measurements

| Experiment                   | Time Seconds | Peak GPU Memory MB | CPU Memory Change MB | Initial Loss | Final Loss | Mean Loss |
| ============================ | ============ | ================== | ==================== | ============ | ========== | ========= |
| Baseline                     | 0.3454       | 1183.7041          | 51.2617              | 0.6917       | 0.6963     | 0.7041    |
| Tensor Data on CPU           | 0.1077       | 1183.7041          | 0.0000               | 0.6917       | 0.6963     | 0.7041    |
| Tensor Data Preloaded to GPU | 0.1009       | 1191.6724          | 0.0000               | 0.6917       | 0.6963     | 0.7041    |
| Default Initialization       | 0.1335       | 1183.7041          | -0.0156              | 0.6917       | 0.6963     | 0.7041    |
| Xavier Initialization        | 0.1284       | 1183.7041          | 0.0000               | 0.7106       | 0.6971     | 0.7321    |
| Checkpointing Disabled       | 0.1207       | 1183.7041          | 0.0000               | 0.6917       | 0.6963     | 0.7041    |
| Activation Checkpointing     | 0.2718       | 1184.7041          | 2.7930               | 0.6917       | 0.6963     | 0.7041    |
| Batch Size 32 Direct         | 0.1242       | 1184.7041          | 0.0000               | 0.6917       | 0.6963     | 0.7041    |
| Gradient Accumulation        | 0.3125       | 1184.6572          | 0.0625               | 0.6917       | 0.6963     | 0.7041    |
| Full Precision FP32          | 0.0999       | 1184.7041          | 0.0000               | 0.6917       | 0.6963     | 0.7041    |
| Mixed Precision              | 0.4314       | 1184.7056          | 84.7930              | 0.6917       | 0.6961     | 0.7043    |

The optimization measurements depend on the hardware used for the run. CUDA available in this run was **True**. I used the timing, memory, and loss values produced by this runtime.
