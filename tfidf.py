#!/usr/bin/env python3
"""TF-IDF — term frequency-inverse document frequency search."""
import math, re, sys
from collections import Counter

class TFIDFEngine:
    def __init__(self):
        self.docs = []; self.vocab = set(); self.idf = {}; self.tfidf = []
    def _tokenize(self, text): return re.findall(r'\w+', text.lower())
    def index(self, documents):
        self.docs = documents; N = len(documents)
        doc_tokens = [self._tokenize(d) for d in documents]
        df = Counter()
        for tokens in doc_tokens:
            for t in set(tokens): df[t] += 1
        self.vocab = set(df.keys())
        self.idf = {t: math.log(N / (1 + df[t])) for t in self.vocab}
        self.tfidf = []
        for tokens in doc_tokens:
            tf = Counter(tokens); n = len(tokens)
            vec = {t: (tf[t]/n) * self.idf.get(t, 0) for t in set(tokens)}
            self.tfidf.append(vec)
    def search(self, query, top_k=5):
        tokens = self._tokenize(query)
        q_tf = Counter(tokens); n = len(tokens)
        q_vec = {t: (q_tf[t]/n) * self.idf.get(t, 0) for t in tokens if t in self.idf}
        scores = []
        for i, dvec in enumerate(self.tfidf):
            dot = sum(q_vec.get(t, 0) * dvec.get(t, 0) for t in q_vec)
            mag_q = math.sqrt(sum(v**2 for v in q_vec.values())) or 1
            mag_d = math.sqrt(sum(v**2 for v in dvec.values())) or 1
            scores.append((dot / (mag_q * mag_d), i))
        scores.sort(reverse=True)
        return [(self.docs[i], s) for s, i in scores[:top_k] if s > 0]

if __name__ == "__main__":
    docs = ["The quick brown fox jumps over the lazy dog",
            "Python is a great programming language",
            "Machine learning with neural networks",
            "The fox and the hound are friends",
            "Deep learning is a subset of machine learning",
            "Programming in Python is fun and productive"]
    engine = TFIDFEngine(); engine.index(docs)
    for q in ["fox", "python programming", "machine learning neural"]:
        results = engine.search(q, 3)
        print(f"Query: '{q}'")
        for doc, score in results: print(f"  [{score:.3f}] {doc}")
        print()
