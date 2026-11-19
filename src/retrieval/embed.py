"""Transformer retrieval: sentence embeddings (BERT-family) + cosine similarity.

Any object with `.encode(list_of_str) -> L2-normalised np.ndarray (n, d)` can be used as
an encoder, so the quantized / ONNX encoders of step 9 plug in without changes.
Constraint handling is identical to the tf-idf baseline: the only difference between the
two systems is how query text and recipe text are matched.
"""
import numpy as np

from src.retrieval.constraints import extract_constraints
from src.retrieval.tfidf import doc_text

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class STEncoder:
    def __init__(self, name=DEFAULT_MODEL, device=None):
        from sentence_transformers import SentenceTransformer
        self.name = name
        self.model = SentenceTransformer(name, device=device)

    def encode(self, texts, batch_size=32):
        return self.model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)


class EmbeddingRetriever:
    def __init__(self, corpus, encoder, use_constraints=True, doc_emb=None):
        self.corpus = corpus.reset_index(drop=True)
        self.encoder, self.use_constraints = encoder, use_constraints
        if doc_emb is None:
            doc_emb = encoder.encode([doc_text(r) for _, r in self.corpus.iterrows()])
        self.E = np.asarray(doc_emb, dtype=np.float32)
        self.ids = self.corpus["recipe_id"].to_numpy()
        self.minutes = self.corpus["minutes"].to_numpy()
        self.diets = self.corpus["diet_tags"].fillna("").to_numpy()

    def scores(self, text):
        q = self.encoder.encode([text])[0]
        return self.E @ q                                    # cosine (both sides normalised)

    def search(self, query, k=5):
        c = extract_constraints(query) if self.use_constraints else None
        text = c.text if (c is not None and c.any) else query
        s = self.scores(text)
        order = np.lexsort((np.arange(len(s)), -s))
        if c is not None and c.any:
            ok = np.array([c.satisfied_by(m, d) for m, d in zip(self.minutes, self.diets)])
            order = np.concatenate([order[ok[order]], order[~ok[order]]])
        k = len(order) if k is None else k
        top = order[:k]
        return top, s[top], c

    def rank_ids(self, query, k=None):
        top, _, _ = self.search(query, k)
        return list(self.ids[top])
