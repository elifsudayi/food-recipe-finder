"""End-to-end Food Photo -> Recipe Finder pipeline (the logic behind the Gradio app).

Image path : photo -> VGG16 fine-tuned, INT8 ONNX (static PTQ)  -> dish + confidence
             photo -> VGG16 fine-tuned, fp32 PyTorch            -> Grad-CAM for the predicted class
Text path  : query -> regex constraint extraction -> MiniLM INT8 ONNX encoder
             -> cosine vs recipe embeddings -> constraint-first ranking -> top-5 recipes
Optional   : the detected dish name is prepended to the text query as context.
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CLASSES, CORPUS_DIR, DISH_NAMES, HF_MODEL_REPO, MODELS_DIR
from src.retrieval.embed import DEFAULT_MODEL, EmbeddingRetriever
from src.retrieval.tfidf import TfidfRetriever, doc_text


def resolve_model_file(filename):
    """Use a local copy (models/onnx/, models/) if present, otherwise download from the Hub."""
    for p in (MODELS_DIR / "onnx" / filename, MODELS_DIR / filename):
        if p.exists():
            return p
    repo = os.environ.get("MODEL_REPO", HF_MODEL_REPO)
    if repo.startswith("YOUR_HF_USERNAME"):
        raise FileNotFoundError(
            f"{filename} not found in models/ and HF_MODEL_REPO is not configured (src/config.py).")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo, filename=filename))


def describe_constraints(c):
    if c is None or not c.any:
        return "none"
    parts = []
    if c.max_minutes is not None:
        parts.append(f"{'<' if c.strict else '≤'} {c.max_minutes:g} min")
    parts += c.diets
    return ", ".join(parts)


class FoodRecipeFinder:
    def __init__(self, class_names, classifier, explainer, corpus, retriever, baseline, transform):
        self.class_names, self.classifier, self.explainer = class_names, classifier, explainer
        self.corpus, self.retriever, self.baseline, self.tf = corpus, retriever, baseline, transform

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls):
        import torch
        from src.image.data import build_transforms
        from src.image.gradcam import GradCAMExplainer
        from src.image.models import build_vgg16
        from src.quant.onnx_utils import OnnxClassifier, OnnxEncoder

        class_names = sorted(CLASSES)                       # same order as torchvision ImageFolder
        classifier = OnnxClassifier(resolve_model_file("vgg16_int8.onnx"))

        vgg = build_vgg16(len(class_names), pretrained=False)
        vgg.load_state_dict(torch.load(resolve_model_file("vgg16_ft.pt"), map_location="cpu"))
        explainer = GradCAMExplainer(vgg, class_names, device="cpu")

        corpus = pd.read_csv(CORPUS_DIR / "recipes.csv")
        corpus["description"] = corpus["description"].fillna("")
        corpus["diet_tags"] = corpus["diet_tags"].fillna("")
        encoder = OnnxEncoder(resolve_model_file("minilm_int8.onnx"), DEFAULT_MODEL, max_len=256)
        retriever = EmbeddingRetriever(corpus, encoder, use_constraints=True)   # embeds recipes once
        baseline = TfidfRetriever(corpus, "stem", True)
        return cls(class_names, classifier, explainer, corpus, retriever, baseline,
                   build_transforms(224, train=False))

    # -------------------------------------------------------------------- image
    def classify(self, pil_img, k=5):
        x = self.tf(pil_img.convert("RGB")).unsqueeze(0).numpy()
        t0 = time.perf_counter()
        probs = self.classifier.probs(x)[0]
        ms = (time.perf_counter() - t0) * 1000
        order = np.argsort(-probs)[:k]
        return {"top": [(self.class_names[i], float(probs[i])) for i in order],
                "pred_idx": int(order[0]), "pred": self.class_names[int(order[0])],
                "confidence": float(probs[order[0]]), "ms": ms}

    def heatmap(self, pil_img, target_idx):
        t0 = time.perf_counter()
        out = self.explainer.explain(pil_img.convert("RGB"), target_class=target_idx)
        out["ms"] = (time.perf_counter() - t0) * 1000
        return out

    # --------------------------------------------------------------------- text
    @staticmethod
    def _table(retr, query, k):
        top, scores, c = retr.search(query, k)
        rows = []
        for rank, (i, s) in enumerate(zip(top, scores), 1):
            r = retr.corpus.iloc[int(i)]
            met = True if (c is None or not c.any) else c.satisfied_by(r["minutes"], r["diet_tags"])
            rows.append({"Rank": rank, "Recipe": r["title"], "Dish": DISH_NAMES.get(r["dish"], r["dish"]),
                         "Time (min)": int(r["minutes"]), "Diet tags": r["diet_tags"].replace("|", ", "),
                         "Similarity": round(float(s), 3), "Constraints": "✓" if met else "✗ not met",
                         "Key ingredients": r["ingredients"][:80]})
        return pd.DataFrame(rows), c, [int(i) for i in top]

    def retrieve(self, query, dish=None, k=5, compare_baseline=False):
        query = (query or "").strip()
        if dish and dish in DISH_NAMES:
            query = f"{DISH_NAMES[dish]} {query}".strip()
        if not query:
            return None
        t0 = time.perf_counter()
        df, c, top = self._table(self.retriever, query, k)
        ms = (time.perf_counter() - t0) * 1000
        r = self.retriever.corpus.iloc[top[0]]
        out = {"query_used": query, "constraints": describe_constraints(c), "table": df, "ms": ms,
               "top_recipe": {"title": r["title"], "minutes": int(r["minutes"]),
                              "ingredients": r["ingredients"].split("; "),
                              "steps": [s for s in r["steps"].split(" || ") if s]},
               "baseline": None}
        if compare_baseline:
            out["baseline"] = self._table(self.baseline, query, k)[0]
        return out
