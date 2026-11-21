"""Gradio app: Food Photo -> Recipe Finder.

Run from the repository root:   python app/app.py          
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import pandas as pd

from src.config import DISH_NAMES
from src.pipeline import FoodRecipeFinder

COLS = ["Rank", "Recipe", "Dish", "Time (min)", "Diet tags", "Similarity", "Constraints", "Key ingredients"]
EMPTY = pd.DataFrame(columns=COLS)

print("Loading models (INT8 VGG16, fp32 VGG16 for Grad-CAM, INT8 MiniLM) ...")
finder = FoodRecipeFinder.load()
print("Ready.")


def identify(img):
    if img is None:
        return None, None, "Please upload a food photo first.", None
    c = finder.classify(img)
    h = finder.heatmap(img, c["pred_idx"])
    info = (f"**Predicted dish: {DISH_NAMES[c['pred']]}** ({c['confidence']:.1%})  \n"
            f"Classifier: VGG16 INT8 · ONNX Runtime · CPU · {c['ms']:.0f} ms  \n"
            f"Grad-CAM: VGG16 fp32 · PyTorch · CPU · {h['ms']:.0f} ms (target = predicted class)")
    if c["confidence"] < 0.5:
        info += "\n\n⚠️ Low confidence – the dish may be misidentified."
    if h["pred"] != c["pred"]:
        info += (f"\n\nNote: the fp32 model predicts *{DISH_NAMES[h['pred']]}*; "
                 f"the heat-map explains the INT8 prediction *{DISH_NAMES[c['pred']]}*.")
    return {DISH_NAMES[n]: p for n, p in c["top"]}, h["overlay"], info, c["pred"]


def find(query, dish, use_dish, compare):
    o = finder.retrieve(query, dish if use_dish else None, k=5, compare_baseline=compare)
    if o is None:
        return "Type a query or upload a photo first.", EMPTY, "", EMPTY
    meta = (f"**Query used:** `{o['query_used']}`  \n**Detected constraints:** {o['constraints']}  \n"
            f"Retrieval: MiniLM INT8 · ONNX Runtime · CPU · {o['ms']:.0f} ms")
    t = o["top_recipe"]
    details = (f"### Top recipe: {t['title']}  (⏱ {t['minutes']} min)\n\n**Ingredients**\n"
               + "\n".join(f"- {i}" for i in t["ingredients"]) + "\n\n**Steps**\n"
               + "\n".join(f"{n}. {s}" for n, s in enumerate(t["steps"], 1)))
    base = o["baseline"] if o["baseline"] is not None else EMPTY
    return meta, o["table"], details, base


with gr.Blocks(title="Food Photo → Recipe Finder") as demo:
    gr.Markdown("# 🍽️ Food Photo → Recipe Finder\n"
                "Upload a food photo to identify the dish (10 classes) and see what the model looked at, "
                "then describe what you want to cook (e.g. *vegetarian, under 30 minutes*).")
    dish_state = gr.State(None)
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="Food photo")
            btn_id = gr.Button("Identify dish", variant="primary")
            label = gr.Label(num_top_classes=5, label="Dish prediction (top-5)")
            info = gr.Markdown()
        with gr.Column():
            cam = gr.Image(label="Grad-CAM heat-map", interactive=False)

    gr.Markdown("## Find recipes")
    with gr.Row():
        query = gr.Textbox(label="What are you looking for?", scale=4,
                           placeholder="e.g. vegetarian, under 30 minutes  /  with mushrooms  /  raw fish and rice")
        use_dish = gr.Checkbox(True, label="Use detected dish as context")
        compare = gr.Checkbox(False, label="Show tf-idf baseline")
    btn_find = gr.Button("Find recipes", variant="primary")
    meta = gr.Markdown()
    table = gr.Dataframe(value=EMPTY, label="Top-5 recipes (MiniLM INT8)", interactive=False, wrap=True)
    details = gr.Markdown()
    base = gr.Dataframe(value=EMPTY, label="tf-idf baseline (comparison)", interactive=False, wrap=True)
    gr.Examples(["quick pizza under 30 minutes", "vegetarian pancakes", "chicken curry with coconut milk",
                 "raw fish and seasoned rice rolled in seaweed", "vegan french fries under 45 minutes"],
                inputs=[query])

    btn_id.click(identify, img, [label, cam, info, dish_state])
    btn_find.click(find, [query, dish_state, use_dish, compare], [meta, table, details, base])
    query.submit(find, [query, dish_state, use_dish, compare], [meta, table, details, base])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    demo.launch(share=ap.parse_args().share)
