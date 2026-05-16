#!/usr/bin/env python3
"""
Question answering using BERT QA model
"""

import tensorflow_hub as hub
from transformers import BertTokenizer
import tensorflow as tf


# Load tokenizer and model once
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)

model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")


def question_answer(question, reference):
    """
    Finds an answer to a question from a reference document

    Args:
        question (str): question to answer
        reference (str): reference document

    Returns:
        str: answer snippet or None if no answer found
    """

    # Tokenize input
    inputs = tokenizer.encode_plus(
        question,
        reference,
        add_special_tokens=True,
        return_tensors="tf",
        truncation=True,
        max_length=512
    )

    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    attention_mask = inputs["attention_mask"]

    # Run model
    outputs = model([
        input_ids,
        attention_mask,
        token_type_ids
    ])

    start_logits, end_logits = outputs

    # Get best start/end positions
    start_idx = tf.argmax(start_logits, axis=1).numpy()[0]
    end_idx = tf.argmax(end_logits, axis=1).numpy()[0]

    # Invalid span
    if end_idx < start_idx:
        return None

    # Convert tokens back to string
    tokens = input_ids[0][start_idx:end_idx + 1]
    answer = tokenizer.decode(tokens)

    # Clean special tokens
    answer = answer.replace("[CLS]", "").replace("[SEP]", "").strip()

    if not answer:
        return None

    return answer
