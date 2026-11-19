"""Classical baseline: TF-IDF + cosine similarity, optional constraint-aware ranking."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.retrieval.constraints import extract_constraints
from src.retrieval.text import TextProcessor


def doc_text(row):
    """The text of a recipe used by ALL retrieval models (title + description + ingredients)."""
    desc = row.get("description", "")
    desc = "" if pd.isna(desc) else str(desc)
    parts = [str(row["title"]), desc, "Ingredients: " + str(row["ingredients"])]
    return ". ".join(p for p in parts if p)


class TfidfRetriever:
    def __init__(self, corpus, mode="stem", use_constraints=False):
        self.corpus = corpus.reset_index(drop=True)
        self.mode, self.use_constraints = mode, use_constraints
        self.proc = TextProcessor(mode)
        self.vec = TfidfVectorizer(analyzer=self.proc)       # default sklearn settings, no tuning
        docs = [doc_text(r) for _, r in self.corpus.iterrows()]
        self.X = self.vec.fit_transform(docs)                # L2-normalised rows
        self.ids = self.corpus["recipe_id"].to_numpy()
        self.minutes = self.corpus["minutes"].to_numpy()
        self.diets = self.corpus["diet_tags"].fillna("").to_numpy()

    def scores(self, text):
        q = self.vec.transform([text])
        return (self.X @ q.T).toarray().ravel()              # cosine similarity

    def search(self, query, k=5):
        """Return (indices, scores, constraints). Recipes satisfying the extracted
        constraints are always ranked above the ones that do not."""
        c = extract_constraints(query) if self.use_constraints else None
        text = c.text if (c is not None and c.any) else query
        s = self.scores(text)
        order = np.lexsort((np.arange(len(s)), -s))          # by score desc, ties by index
        if c is not None and c.any:
            ok = np.array([c.satisfied_by(m, d) for m, d in zip(self.minutes, self.diets)])
            order = np.concatenate([order[ok[order]], order[~ok[order]]])
        k = len(order) if k is None else k
        top = order[:k]
        return top, s[top], c

    def rank_ids(self, query, k=None):
        top, _, _ = self.search(query, k)
        return list(self.ids[top])
