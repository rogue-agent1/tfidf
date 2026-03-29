#!/usr/bin/env python3
"""tfidf - TF-IDF vectorizer and cosine similarity."""
import sys, math, re
from collections import Counter

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def tf(doc):
    tokens = tokenize(doc) if isinstance(doc, str) else doc
    counts = Counter(tokens)
    total = len(tokens)
    return {t: c/total for t, c in counts.items()}

def idf(corpus):
    n = len(corpus)
    doc_freq = Counter()
    for doc in corpus:
        tokens = set(tokenize(doc) if isinstance(doc, str) else doc)
        for t in tokens:
            doc_freq[t] += 1
    return {t: math.log(n / df) for t, df in doc_freq.items()}

def tfidf(corpus):
    idf_scores = idf(corpus)
    vectors = []
    for doc in corpus:
        tf_scores = tf(doc)
        vec = {t: tf_scores[t] * idf_scores.get(t, 0) for t in tf_scores}
        vectors.append(vec)
    return vectors, idf_scores

def cosine_similarity(a, b):
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v**2 for v in a.values()))
    nb = math.sqrt(sum(v**2 for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def most_similar(query, corpus, top_k=3):
    vecs, idf_scores = tfidf(corpus + [query])
    q_vec = vecs[-1]
    scores = [(i, cosine_similarity(q_vec, vecs[i])) for i in range(len(corpus))]
    return sorted(scores, key=lambda x: -x[1])[:top_k]

def test():
    corpus = [
        "the cat sat on the mat",
        "the dog played in the park",
        "cats and dogs are friends",
    ]
    vecs, idf_scores = tfidf(corpus)
    assert len(vecs) == 3
    assert "the" in idf_scores
    # "the" appears in all docs, low IDF
    assert idf_scores["the"] < idf_scores.get("cat", 1)
    # similarity
    sim = cosine_similarity(vecs[0], vecs[2])
    assert 0 <= sim <= 1
    # search
    results = most_similar("cat on mat", corpus)
    assert results[0][0] == 0  # most similar to first doc
    print("OK: tfidf")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        print("Usage: tfidf.py test")
