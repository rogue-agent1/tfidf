from tfidf import tf, idf, tfidf, cosine_similarity, top_terms
t = tf("the cat the dog")
assert abs(t["the"] - 0.5) < 0.01
assert abs(t["cat"] - 0.25) < 0.01
docs = ["cat dog", "cat bird", "dog fish"]
idf_scores = idf(docs)
assert idf_scores["cat"] < idf_scores["fish"]  # cat in 2 docs, fish in 1
scores = tfidf(docs)
assert len(scores) == 3
assert "cat" in scores[0]
sim = cosine_similarity({"a":1,"b":2}, {"a":1,"b":2})
assert abs(sim - 1.0) < 0.01
sim2 = cosine_similarity({"a":1}, {"b":1})
assert abs(sim2) < 0.01
top = top_terms(scores[2], 2)
assert len(top) == 2
print("tfidf tests passed")
