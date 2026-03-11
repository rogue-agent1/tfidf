#!/usr/bin/env python3
"""TF-IDF text vectorizer."""
import sys, math, re, collections
def tokenize(text): return re.findall(r'\w+',text.lower())
def tfidf(docs):
    tf=[collections.Counter(tokenize(d)) for d in docs]
    vocab=sorted(set(w for c in tf for w in c))
    df={w:sum(1 for c in tf if w in c) for w in vocab}
    n=len(docs); idf={w:math.log(n/(df[w]+1))+1 for w in vocab}
    vectors=[]
    for c in tf:
        total=sum(c.values())
        vectors.append({w:c[w]/total*idf[w] for w in vocab if w in c})
    return vectors,vocab
docs=["the cat sat on the mat","the dog sat on the log","cats and dogs are friends"]
vecs,vocab=tfidf(docs)
print("TF-IDF Vectors:")
for i,d in enumerate(docs):
    top=sorted(vecs[i].items(),key=lambda x:-x[1])[:5]
    print(f"  Doc {i}: {top}")
