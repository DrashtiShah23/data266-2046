# AI Use Appendix

## 1. What I used an assistant for

I used ChatGPT mainly to help organize the notebook, work through errors, and understand how to connect the Word2Vec, RAG, and optimization parts of the homework. I used some of the code as a starting point, but I ran everything myself in Colab and checked the outputs before keeping it.

I also checked the nearest neighbor results, the retrieved passages for the RAG questions, and the timing, memory, and loss values from the optimization experiments.

## 2. One specific thing the assistant got wrong

The first version tried to load IMDB using the Hugging Face datasets library.

The code looked correct, but it failed in my Colab runtime with an HfUriError. I also ran into another problem with the original Wikipedia loader, which returned a JSON decoding error while trying to load the first movie.

## 3. How I noticed the problem

I found both problems while running the notebook from the top.

The Word2Vec model loaded successfully, but execution stopped when IMDB was loaded. Since the dataset was never created, the tokenization and fine tuning cells could not continue.

The Wikipedia error happened before any movie documents were created, which showed me that the issue was with the loader rather than the rest of the RAG pipeline.

## 4. What I changed and why it works

For IMDB, I switched to the Keras IMDB dataset and decoded the integer sequences back into review text before using the reviews for Word2Vec fine tuning.

For the RAG section, I replaced the failing Wikipedia loader with LangChain WebBaseLoader and used the direct Wikipedia pages for the ten movies.

After making the changes, I reran those sections and checked that the reviews and all ten movie documents loaded correctly before continuing with the rest of the assignment.
