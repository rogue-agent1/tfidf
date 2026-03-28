#!/usr/bin/env python3
"""tfidf - TF-IDF text analysis from scratch."""
import argparse, math, re, json
from collections import Counter

def tokenize(text: str) -> list:
    return re.findall(r'\b\w+\b', text.lower())

def tf(doc: list) -> dict:
    counts = Counter(doc)
    total = len(doc)
    return {w: c / total for w, c in counts.items()}

def idf(docs: list) -> dict:
    n = len(docs)
    df = Counter()
    for doc in docs:
        df.update(set(doc))
    return {w: math.log(n / (1 + count)) for w, count in df.items()}

def tfidf(docs: list) -> list:
    idf_scores = idf(docs)
    results = []
    for doc in docs:
        tf_scores = tf(doc)
        results.append({w: tf_scores[w] * idf_scores.get(w, 0) for w in tf_scores})
    return results

def main():
    p = argparse.ArgumentParser(description="TF-IDF analysis")
    p.add_argument("files", nargs="+")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    docs = [tokenize(open(f).read()) for f in args.files]
    scores = tfidf(docs)
    for i, (f, s) in enumerate(zip(args.files, scores)):
        top = sorted(s.items(), key=lambda x: -x[1])[:args.top]
        if args.json:
            print(json.dumps({"file": f, "terms": [{"term": t, "score": round(sc, 4)} for t, sc in top]}))
        else:
            print(f"\n=== {f} ===")
            for t, sc in top:
                print(f"  {sc:.4f} {t}")

if __name__ == "__main__":
    main()
