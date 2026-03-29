#!/usr/bin/env python3
"""TF-IDF vectorizer and document similarity."""
import sys, math, re
from collections import Counter

class TfIdf:
    def __init__(self):
        self.idf = {}
        self.vocab = set()
        self.doc_count = 0
    def tokenize(self, text):
        return re.findall(r"[a-z]+", text.lower())
    def fit(self, documents):
        self.doc_count = len(documents)
        df = Counter()
        for doc in documents:
            tokens = set(self.tokenize(doc))
            self.vocab |= tokens
            for t in tokens: df[t] += 1
        self.idf = {t: math.log(self.doc_count / (1 + df[t])) for t in self.vocab}
    def transform(self, doc):
        tokens = self.tokenize(doc)
        tf = Counter(tokens)
        n = len(tokens) or 1
        return {t: (tf[t]/n) * self.idf.get(t, 0) for t in set(tokens) if t in self.idf}
    def cosine_sim(self, v1, v2):
        common = set(v1) & set(v2)
        if not common: return 0
        dot = sum(v1[k]*v2[k] for k in common)
        n1 = math.sqrt(sum(v*v for v in v1.values()))
        n2 = math.sqrt(sum(v*v for v in v2.values()))
        return dot/(n1*n2) if n1 and n2 else 0

def test():
    docs = ["the cat sat on the mat", "the dog sat on the rug", "birds fly in the sky"]
    tfidf = TfIdf()
    tfidf.fit(docs)
    v1 = tfidf.transform(docs[0])
    v2 = tfidf.transform(docs[1])
    v3 = tfidf.transform(docs[2])
    sim12 = tfidf.cosine_sim(v1, v2)
    sim13 = tfidf.cosine_sim(v1, v3)
    assert sim12 > sim13, f"cat/dog docs should be more similar: {sim12} vs {sim13}"
    assert "cat" in v1
    assert tfidf.idf.get("the", 0) < tfidf.idf.get("cat", 1)  # "the" is common
    print("  tfidf: ALL TESTS PASSED")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test": test()
    else: print("TF-IDF vectorizer")
