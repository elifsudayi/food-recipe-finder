# Food Photo → Recipe Finder

SWE015 – Building an Intelligent Application · **Use Case 3** · Elif Su Dayı (2309011032)

A small cooking assistant. The user uploads a **photo of a dish** and gets the predicted dish with a confidence
score and a **Grad-CAM heat-map**; then the user types a free-text request such as
*"vegetarian, under 30 minutes"* and gets the **top-5 recipes with similarity scores**.

The application runs **INT8-quantized models** on a CPU:

| Part | Model that runs in the app | Precision |
|---|---|---|
| Dish classification | VGG16 (ImageNet weights, fine-tuned on 10 Food-101 classes) | **INT8** (ONNX Runtime) |
| Recipe retrieval | `all-MiniLM-L6-v2` sentence encoder (BERT-based Transformer) | **INT8** (ONNX Runtime) |
| Grad-CAM explanation | the same fine-tuned VGG16 | fp32 (PyTorch), explanation only |
| Comparison (optional checkbox) | tf-idf + cosine baseline | – |

Architecture diagram: [`docs/architecture.png`](docs/architecture.png)

## Quick start

```bash
git clone https://github.com/elifsudayi/food-recipe-finder.git
cd food-recipe-finder
pip install -r requirements.txt
python app/app.py
```

Then open the address printed in the terminal (by default <http://127.0.0.1:7860>).
Add `--share` (`python app/app.py --share`) to get a public link, e.g. when running in Google Colab.

* **Python 3.10 or newer** is recommended. The project was developed and tested on Google Colab.
* **First start needs internet.** The model weights (~160 MB in total) and the MiniLM tokenizer are downloaded
  from the Hugging Face Hub and cached; start-up takes a few minutes the first time and a few seconds afterwards.
* No GPU is needed; the app runs on CPU.

### Model weights

Large weight files are **not** stored in this repository. They are hosted on the Hugging Face Hub:

**<https://huggingface.co/elifsudayi/food-recipe-finder-models>**

| File | Size | Used for |
|---|---|---|
| `vgg16_int8.onnx` | 27.7 MB | dish classification (INT8) |
| `vgg16_ft.pt` | 110.3 MB | Grad-CAM (fp32 fine-tuned VGG16) |
| `minilm_int8.onnx` | 22.8 MB | recipe retrieval (INT8) |

The app looks for the files in `models/onnx/` and `models/` first and downloads them only if they are missing.
The repository name is set in `src/config.py` (`HF_MODEL_REPO`) and can be overridden with the environment
variable `MODEL_REPO`.

## Using the app

1. Upload a food photo and press **Identify dish** → top-5 probabilities, Grad-CAM heat-map and timings.
2. Type a request in **What are you looking for?** (e.g. `vegetarian, under 30 minutes`, `with mushrooms`,
   `raw fish and seasoned rice rolled in seaweed`) and press **Find recipes**.
   * *Use detected dish as context* prepends the predicted dish name to the request.
   * *Show tf-idf baseline* shows the classical baseline's results below the main table.
   * Time limits and diet tags (vegetarian, vegan, gluten-free, dairy-free, low-carb, low-fat, …) are extracted with
     regular expressions; recipes that satisfy them are ranked first, and the *Constraints* column shows ✓ / ✗.

The app works on the 10 dishes it was trained on: caesar salad, chicken curry, french fries, hamburger, ice cream,
omelette, pancakes, pizza, spaghetti carbonara, sushi. For any other dish it still has to choose one of these
ten, and a warning appears when its confidence is below 50 %.

## Results

All numbers come from my own runs; the raw measurements are in [`results/`](results/).

The final latencies in the tables come from results/quant_vgg16_metrics.json and results/quant_minilm_metrics.json, where fp32 and INT8 models were timed in the same session on the same CPU. The latency fields in the earlier per-model JSON files (small_cnn_metrics.json, vgg16_metrics.json, retrieval_*_metrics.json) are preliminary measurements from other Colab machines and differ slightly.

**Image models** – 2,500 held-out test photos (250 per class). Latency: mean of 100 single-image runs on CPU
(AMD EPYC 7B12, 2 vCPU, Google Colab).

| Model | Top-1 | Top-5 | Size (MB) | Latency (ms) |
|---|---|---|---|---|
| Small CNN from scratch, fp32 | 80.2 % | 98.1 % | 1.58 | 12.3 |
| VGG16 feature extraction, fp32 | 89.8 % | 99.4 % | 110.27 | 404.7 ¹ |
| VGG16 fine-tuned, fp32 (PyTorch) | 92.1 % | 99.6 % | 110.27 | 404.7 |
| **VGG16 fine-tuned, INT8 (ONNX Runtime)** | 91.0 % | 99.3 % | 27.67 | 205.3 |

¹ Same architecture as the fine-tuned model; latency measured once. INT8 predictions agree with fp32 on 96.2 % of
the test photos.

**Retrieval models** – 40 labelled queries. Latency: mean of 200 single-query runs on CPU (Intel Xeon 2.20 GHz,
2 vCPU, Google Colab).

| Model | MRR | Recall@5 | Size (MB) | Latency (ms) |
|---|---|---|---|---|
| tf-idf + constraints (classical baseline) | 0.946 | 0.855 | 0.16 | 1.4 (full search) |
| MiniLM, fp32 (PyTorch) | 0.938 | 0.883 | 90.89 | 20.9 (query encoding) |
| **MiniLM, INT8 (ONNX Runtime)** | 0.951 | 0.889 | 22.84 | 5.4 (query encoding), 5.6 (full search) |

Recall@5 = |relevant ∩ top-5| / min(5, |relevant|). With only 40 queries, the differences between the three
retrieval models are small and should not be over-interpreted. Quantization shrank both big models about 4× and made
them 2× (VGG16) and 3.9× (MiniLM encoder) faster than their PyTorch fp32 versions.

## Data and sources

* **Images:** [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (Bossard et al., ETH Zürich, 2014),
  10 classes. Splits: train 650 / validation 100 / test 250 images per class (validation drawn from the official
  train split with seed 42; test = official test split).
* **Recipes:** *Food.com Recipes and Interactions* on Kaggle (`shuyangli94/food-com-recipes-and-user-interactions`),
  file `RAW_recipes.csv`. 300 recipes (30 per dish) were selected by name patterns and basic quality filters. Please
  check the dataset page for its license and terms. The selected subset is in [`corpus/recipes.csv`](corpus/recipes.csv).
* **Test queries:** [`corpus/queries.csv`](corpus/queries.csv) – 40 queries of five types (time, diet, ingredient,
  time+diet, semantic paraphrase). Relevant recipes are defined by the documented criteria in the `criteria` column
  (recipe metadata / ingredient lists), not by any model's output.
* **Models and methods:** VGG16 (Simonyan & Zisserman, 2015), Grad-CAM (Selvaraju et al., 2017),
  `sentence-transformers/all-MiniLM-L6-v2`, ONNX Runtime.

## Repository structure

```
app/app.py                  Gradio application
src/pipeline.py             end-to-end logic used by the app
src/config.py               classes, paths, Hugging Face repo name
src/data/make_food10.py     builds the 10-class Food-101 subset and its splits
src/image/                  data loading + augmentation, models (small CNN, VGG16), training loop, Grad-CAM
src/retrieval/              recipe corpus, query generation, text processing, regex constraints,
                            tf-idf retriever, embedding retriever, metrics
src/quant/onnx_utils.py     ONNX export, INT8 quantization, ONNX Runtime wrappers, latency helper
corpus/                     recipes.csv, queries.csv
results/                    metrics (JSON) and figures (Grad-CAM, confusion matrix, splits)
docs/                       architecture diagram (+ script that draws it)
notebooks/                  Colab notebooks used for training, evaluation and quantization
```

## Reproducing the experiments

Training and evaluation were run in Google Colab (T4 GPU for training, CPU for benchmarks). The notebooks in
`notebooks/` were run in this order:

| Notebook | What it does | Output |
|---|---|---|
| `01_setup` | environment and GPU check | – |
| `02_data_food10` | downloads Food-101, builds the 10-class splits | `results/splits.csv` |
| `03_small_cnn` | small CNN from scratch | `results/small_cnn_metrics.json` |
| `04_vgg16_transfer` | VGG16 feature extraction → fine-tuning | `results/vgg16_metrics.json` |
| `05_gradcam` | Grad-CAM examples, confusion matrix | `results/gradcam_*.png`, `confusion_matrix.png` |
| `06_corpus_queries` | builds the recipe corpus and labelled queries | `corpus/*.csv` |
| `07_tfidf_baseline` | query processing + tf-idf baseline | `results/retrieval_tfidf_metrics.json` |
| `08_embedding_retrieval` | MiniLM retrieval | `results/retrieval_minilm_metrics.json` |
| `09a_quantize_vgg16` | VGG16 → INT8 ONNX, benchmark | `results/quant_vgg16_metrics.json` |
| `09b_quantize_minilm` | MiniLM → INT8 ONNX, benchmark | `results/quant_minilm_metrics.json` |
| `10_app_test` | runs the app in Colab | – |

Seeds are fixed (`SEED = 42` in `src/config.py`). Latencies depend on the machine, so a rerun will give different
milliseconds; accuracies should be close to the values above.

## Limitations

* Only 10 dishes, 300 recipes and 40 test queries; retrieval differences are within noise.
* Relevance labels are derived from recipe metadata, so the gain from constraint extraction is partly by construction,
  and "semantic" queries are judged at the dish level.
* Grad-CAM is computed on the fp32 model, not on the INT8 model.
* Latency was measured on shared 2-vCPU Colab machines, and the image and text benchmarks ran on different CPUs.

## AI usage

Claude (Anthropic) was used as a coding and writing assistant for code, debugging, experiment design and drafting
the slides. I ran all notebooks myself and every number in this repository comes from my own runs.
