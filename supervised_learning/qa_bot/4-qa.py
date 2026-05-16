#!/usr/bin/env python3
"""
Question answering across multiple documents
"""

semantic_search = __import__('3-semantic_search').semantic_search
qa = __import__('0-qa').question_answer


def question_answer(corpus_path):
    """
    Interactive QA loop using semantic search + BERT QA

    Args:
        corpus_path (str): path to corpus directory
    """

    exit_words = ["exit", "quit", "goodbye", "bye"]

    while True:
        question = input("Q: ")

        if question.lower() in exit_words:
            print("A: Goodbye")
            break

        # Find most relevant document
        reference = semantic_search(corpus_path, question)

        # Extract answer
        answer = qa(question, reference)

        if answer is None or answer.strip() == "":
            print("A: Sorry, I do not understand your question.")
        else:
            print("A:", answer)
