#!/usr/bin/env python3
"""TF-IDF text similarity from scratch."""
import sys, math, re
from collections import Counter

def tokenize(text): return re.findall(r'\w+', text.lower())

def tf(doc):
    words = tokenize(doc); n = len(words)
    return {w: c/n for w, c in Counter(words).items()}

def idf(docs):
    n = len(docs); df = Counter()
    for doc in docs:
        for w in set(tokenize(doc)): df[w] += 1
    return {w: math.log(n/(c+1))+1 for w, c in df.items()}

def tfidf(docs):
    idf_scores = idf(docs)
    vectors = []
    for doc in docs:
        tf_scores = tf(doc)
        vectors.append({w: tf_scores[w]*idf_scores.get(w,1) for w in tf_scores})
    return vectors

def cosine_sim(a, b):
    keys = set(a) | set(b)
    dot = sum(a.get(k,0)*b.get(k,0) for k in keys)
    na = math.sqrt(sum(v**2 for v in a.values()))
    nb = math.sqrt(sum(v**2 for v in b.values()))
    return dot/(na*nb) if na and nb else 0

if __name__ == '__main__':
    if '--demo' in sys.argv:
        docs = ["the cat sat on the mat","the dog sat on the log","cats and dogs are friends",
                "machine learning is fun","deep learning neural networks","AI and machine learning"]
        vectors = tfidf(docs)
        print("Document similarity matrix:\n")
        for i in range(len(docs)):
            sims = [f"{cosine_sim(vectors[i],vectors[j]):.2f}" for j in range(len(docs))]
            print(f"  D{i}: {' '.join(sims)}  '{docs[i][:30]}'")
    elif len(sys.argv) > 2:
        query = sys.argv[1]
        docs = sys.argv[2:]
        all_docs = [query] + docs
        vectors = tfidf(all_docs)
        print(f"Query: '{query}'\n\nRanked results:")
        ranked = sorted(range(1,len(all_docs)), key=lambda i: cosine_sim(vectors[0],vectors[i]), reverse=True)
        for i in ranked:
            print(f"  {cosine_sim(vectors[0],vectors[i]):.3f}  '{docs[i-1][:50]}'")
    else:
        print("Usage: tfidf.py --demo | tfidf.py 'query' 'doc1' 'doc2' ...")
