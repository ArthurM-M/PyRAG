# PyRag

A Python implementation of a RAG system. This project integrates the Google Gemini API with a hybrid retrieval via BM25 and vector embedding.

## What's RAG?

Retrieval-Augmented Generation (RAG) is an architecture that combines information retrieval with Large Language Models (LLMs). Instead of relying only on the model's training data, a RAG system retrieves relevant information from an external knowledge source and includes it in the prompt before generating a response.

A typical RAG pipeline consists of three steps:
1. Retrieve relevant documents from a knowledge base.
2. Provide the retrieved context to the language model.
3. Generate a response based on both the user query and the retrieved information.

## Why Use RAG?

RAG addresses several limitations of standalone language models:
* **Reduces hallucinations** by grounding responses in external data.
* **Allows access to information** that was not present during model training.
* **Keeps knowledge up to date** without retraining the model.
* **Improves transparency** by allowing retrieved sources to be inspected.

## What kind of problems can solved with RAG?

RAG can be useful anywhere there is specific information that isn't in a LLM training data.

For example:
* Customer support chat
* Company internal documentation chat
* Textbook Q&A

## What's BM25?

BM25 (Best Matching 25) is a ranking algorithm used in information retrieval systems and search engines.

It scores documents based on:
* The **frequency** of query terms within a document.
* The **rarity** of those terms across the entire collection.
* The **length** of the document.

BM25 relies on exact term matching. This makes it effective when the query contains important keywords that must be present in the retrieved documents.

## What's vector embedding?

While BM25 looks for exact words, Vector Embeddings allow the system to understand the meaning behind those words. An embedding is a mathematical representation of text converted into a long string of numbers (a vector) by a machine learning model.

Words or sentences with similar meanings are placed close to each other in a mathematical space. For example, the vectors for "king" and "queen", or "Python" and "programming", will be tightly clustered. This allows the RAG system to find relevant documents even if the user uses completely different words than the source text (e.g., searching for "feline illnesses" and retrieving a document about "sick cats").

## Why use hybrid retrieval?

By merging keyword precision with semantic depth, hybrid retrieval ensures the LLM receives the most accurate context possible.

