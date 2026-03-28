#!/usr/bin/env python3
"""tfidf - TF-IDF text analysis for document comparison."""
import sys, re, math, collections

STOP = {'the','a','an','is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might','shall','can',
    'in','on','at','to','for','of','with','by','from','as','into','through','during',
    'and','but','or','nor','not','so','yet','both','either','neither','this','that',
    'these','those','it','its','i','you','he','she','we','they','me','him','her','us','them'}

def tokenize(text):
    return [w for w in re.findall(r'\b\w+\b', text.lower()) if w not in STOP and len(w) > 2]

def tf(doc):
    words = tokenize(doc)
    counts = collections.Counter(words)
    total = len(words)
    return {w: c/total for w, c in counts.items()}

def idf(docs):
    n = len(docs)
    df = collections.Counter()
    for doc in docs:
        df.update(set(tokenize(doc)))
    return {w: math.log(n / (1+c)) for w, c in df.items()}

def tfidf(docs, top=10):
    idf_scores = idf(docs)
    results = []
    for i, doc in enumerate(docs):
        tf_scores = tf(doc)
        scores = {w: tf_scores[w] * idf_scores.get(w, 0) for w in tf_scores}
        top_words = sorted(scores.items(), key=lambda x: -x[1])[:top]
        results.append(top_words)
    return results

def main():
    args = sys.argv[1:]
    if not args or '-h' in args:
        print("Usage: tfidf.py FILE1 FILE2 ... [--top N]"); return
    top = int(args[args.index('--top')+1]) if '--top' in args else 10
    files = [a for a in args if not a.startswith('--') and a != str(top)]
    docs = [open(f).read() for f in files]
    results = tfidf(docs, top)
    for i, (fname, words) in enumerate(zip(files, results)):
        print(f"\n📄 {fname}:")
        for w, score in words:
            bar = '█' * int(score * 200)
            print(f"  {score:.4f} {bar} {w}")

if __name__ == '__main__': main()
