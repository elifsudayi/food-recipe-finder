"""Draws docs/architecture.png (matches src/pipeline.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C_INT8, C_FP32, C_RULE, C_IO, C_BASE = "#F6C177", "#9CC5E8", "#D9D9D9", "#FFFFFF", "#EFEFEF"

fig, ax = plt.subplots(figsize=(16, 9.2))
ax.set_xlim(0, 160); ax.set_ylim(0, 92); ax.axis("off")


def box(x, y, w, h, text, color, bold=False, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=color, ec="#333333", lw=1.2, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(p, q, dashed=False, label=None, lpos=None, via=None):
    pts = [p] + (via or []) + [q]
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#333333", lw=1.4, ls="--" if dashed else "-")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.4, ls="--" if dashed else "-"))
    if label:
        ax.text(*lpos, label, fontsize=8.2, style="italic", color="#444444", ha="center")


ax.text(80, 90.3, "Food Photo → Recipe Finder — runtime architecture (Gradio app)", ha="center",
        fontsize=14, fontweight="bold")

# ---- image path ----
box(3, 58, 17, 12, "Food photo\n(upload)", C_IO)
box(26, 58, 25, 12, "Preprocess\nresize 255 · center-crop 224\nImageNet normalisation", C_RULE)
box(57, 58, 30, 12, "VGG16 fine-tuned\nINT8 ONNX (static PTQ)\nONNX Runtime · CPU", C_INT8, bold=True)
box(93, 58, 26, 12, "Dish + confidence\ntop-5 probabilities", C_IO)
box(57, 76, 30, 11, "Grad-CAM\nVGG16 fine-tuned fp32 (PyTorch)\nlast conv layer (conv5_3)", C_FP32)
box(93, 76, 26, 11, "Heat-map overlay\non the model's 224×224 crop", C_IO)
arrow((20.4, 64), (25.6, 64)); arrow((51.4, 64), (56.6, 64)); arrow((87.4, 64), (92.6, 64))
arrow((38.5, 70.4), (56.6, 81.5), via=[(38.5, 81.5)])
arrow((72, 70.4), (72, 75.6), label="target = predicted class", lpos=(87, 73.2))
arrow((87.4, 81.5), (92.6, 81.5))

# ---- text path ----
box(3, 20, 17, 12, "Text query\n(+ optional dish name)", C_IO)
box(26, 20, 28, 12, "Constraint extraction\nregex: time (≤ / < N min)\nand diet tags → cleaned text", C_RULE)
box(60, 20, 32, 12, "MiniLM encoder (all-MiniLM-L6-v2)\nINT8 ONNX (dynamic PTQ) · CPU\ntokenise → transformer → pool → L2", C_INT8, bold=True)
box(98, 20, 30, 12, "Cosine similarity vs. 300\nrecipe embeddings\n+ constraint-first ranking", C_RULE)
box(134, 20, 23, 12, "Top-5 recipes\nwith similarity\nscores + top recipe", C_IO)
arrow((20.4, 26), (25.6, 26)); arrow((54.4, 26), (59.6, 26)); arrow((92.4, 26), (97.6, 26)); arrow((128.4, 26), (133.6, 26))
box(98, 4, 30, 10, "tf-idf (stemming) baseline\nshown only for comparison", C_BASE, dashed=True)
arrow((113, 14.4), (113, 19.6), dashed=True)

# ---- link between the two paths ----
arrow((106, 57.6), (11.5, 32.6), dashed=True, via=[(106, 44), (11.5, 44)])
ax.text(60, 45.2, "detected dish name used as optional context for the text query", fontsize=8.6,
        style="italic", color="#444444", ha="center")

# ---- offline + legend ----
ax.text(3, 10.5, "Offline (notebooks):", fontsize=9.2, fontweight="bold")
ax.text(3, 7.2, "Food-101 (10 classes) → small CNN · VGG16 feature extraction → fine-tuning → INT8 PTQ", fontsize=8.6)
ax.text(3, 4.0, "Food.com recipes (300) + 40 labelled queries → tf-idf · MiniLM → INT8 PTQ", fontsize=8.6)
for i, (c, t) in enumerate([(C_INT8, "INT8 quantized model (runs the predictions)"), (C_FP32, "fp32 model (explanation only)"),
                            (C_RULE, "rule-based / numeric step"), (C_IO, "input / output")]):
    ax.add_patch(FancyBboxPatch((134, 13 - i * 3.1), 3.2, 2.0, boxstyle="round,pad=0.2", fc=c, ec="#333333", lw=1))
    ax.text(139, 14 - i * 3.1, t, fontsize=8.4, va="center")

plt.savefig("docs/architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
