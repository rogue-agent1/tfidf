#!/usr/bin/env python3
"""tfidf - TF-IDF text vectorizer with cosine similarity search."""
import sys, json, math, re
from collections import Counter

def tokenize(text):
    return re.findall(r'\b[a-z]+\b', text.lower())

class TfIdf:
    def __init__(self):
        self.docs = []; self.df = Counter(); self.vocab = set()
    
    def add(self, text):
        tokens = tokenize(text)
        self.docs.append(tokens)
        for t in set(tokens): self.df[t] += 1
        self.vocab.update(tokens)
    
    def tfidf(self, doc_idx):
        tokens = self.docs[doc_idx]; tf = Counter(tokens); n = len(self.docs)
        vec = {}
        for t, count in tf.items():
            tf_val = count / len(tokens)
            idf_val = math.log(n / (self.df[t] + 1)) + 1
            vec[t] = tf_val * idf_val
        return vec
    
    def cosine_sim(self, v1, v2):
        common = set(v1.keys()) & set(v2.keys())
        dot = sum(v1[k]*v2[k] for k in common)
        n1 = math.sqrt(sum(v**2 for v in v1.values()))
        n2 = math.sqrt(sum(v**2 for v in v2.values()))
        return dot/(n1*n2) if n1 and n2 else 0
    
    def search(self, query, top_k=3):
        qtokens = tokenize(query)
        qtf = Counter(qtokens); n = len(self.docs)
        qvec = {t: (c/len(qtokens))*(math.log(n/(self.df.get(t,0)+1))+1) for t, c in qtf.items()}
        scores = []
        for i in range(len(self.docs)):
            dvec = self.tfidf(i)
            scores.append((i, self.cosine_sim(qvec, dvec)))
        return sorted(scores, key=lambda x: -x[1])[:top_k]

def main():
    tfidf = TfIdf()
    docs = [
        "the cat sat on the mat",
        "the dog played in the park",
        "cats and dogs are popular pets",
        "machine learning is a subset of artificial intelligence",
        "deep learning uses neural networks",
        "natural language processing handles text data",
    ]
    for d in docs: tfidf.add(d)
    print("TF-IDF demo\n")
    results = tfidf.search("neural network machine learning")
    for idx, score in results:
        print(f"  [{score:.3f}] {docs[idx]}")
    v0 = tfidf.tfidf(0)
    top_terms = sorted(v0.items(), key=lambda x: -x[1])[:5]
    print(f"\n  Doc 0 top terms: {[(t, round(s,3)) for t,s in top_terms]}")

if __name__ == "__main__":
    main()
