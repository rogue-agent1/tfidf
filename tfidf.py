#!/usr/bin/env python3
"""TF-IDF calculator. Zero dependencies."""
import math, re
from collections import Counter

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def tf(doc):
    words = tokenize(doc)
    n = len(words)
    return {w: c/n for w, c in Counter(words).items()} if n else {}

def idf(docs):
    n = len(docs)
    doc_freq = Counter()
    for doc in docs:
        doc_freq.update(set(tokenize(doc)))
    return {w: math.log(n / (df + 1)) + 1 for w, df in doc_freq.items()}

def tfidf(docs):
    idf_scores = idf(docs)
    result = []
    for doc in docs:
        tf_scores = tf(doc)
        result.append({w: tf_scores[w] * idf_scores.get(w, 0) for w in tf_scores})
    return result

def cosine_similarity(v1, v2):
    keys = set(v1) | set(v2)
    dot = sum(v1.get(k,0)*v2.get(k,0) for k in keys)
    n1 = math.sqrt(sum(v**2 for v in v1.values()))
    n2 = math.sqrt(sum(v**2 for v in v2.values()))
    return dot/(n1*n2) if n1*n2 else 0

def top_terms(tfidf_doc, n=10):
    return sorted(tfidf_doc.items(), key=lambda x: x[1], reverse=True)[:n]

if __name__ == "__main__":
    docs = ["the cat sat on the mat", "the dog sat on the log", "cats and dogs"]
    scores = tfidf(docs)
    for i, s in enumerate(scores):
        print(f"Doc {i}: {top_terms(s, 3)}")
