#!/usr/bin/env python3
"""
Semantic search using Universal Sentence Encoder
"""

import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


# Load model once
model = hub.load(
    "https://tfhub.dev/google/universal-sentence-encoder-large/5"
)


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents

    Args:
        corpus_path (str): path to corpus directory
        sentence (str): query sentence

    Returns:
        str: most semantically similar document text
    """

    documents = []
    doc_texts = []

    # Read all files in corpus
    for filename in os.listdir(corpus_path):
        filepath = os.path.join(corpus_path, filename)

        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(text)
                doc_texts.append(text)

    # Embed query + documents
    embeddings = model([sentence] + doc_texts)

    query_embedding = embeddings[0]
    doc_embeddings = embeddings[1:]

    # Compute cosine similarity
    similarities = []

    for doc_embedding in doc_embeddings:
        similarity = tf.keras.losses.cosine_similarity(
            query_embedding,
            doc_embedding,
            axis=0
        )

        similarities.append(-similarity.numpy())

    # Get best matching document
    best_idx = np.argmax(similarities)

    return documents[best_idx]
