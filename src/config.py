"""Central configuration. Every script/notebook imports from here."""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
CORPUS_DIR = ROOT / "corpus"   # recipes.csv + queries.csv (small, committed to the repo)

# 10-class subset of Food-101
CLASSES = [
    "pizza", "sushi", "hamburger", "ice_cream", "caesar_salad",
    "spaghetti_carbonara", "pancakes", "omelette", "french_fries", "chicken_curry",
]

IMG_SIZE = 224

# Human-readable dish names (used to build queries and for the UI)
DISH_NAMES = {
    "pizza": "pizza", "sushi": "sushi", "hamburger": "hamburger", "ice_cream": "ice cream",
    "caesar_salad": "caesar salad", "spaghetti_carbonara": "spaghetti carbonara",
    "pancakes": "pancakes", "omelette": "omelette", "french_fries": "french fries",
    "chicken_curry": "chicken curry",
}

# Hugging Face Hub model repo that holds the large weights (INT8 ONNX models + fp32 VGG16 for Grad-CAM).
HF_MODEL_REPO = "elifsudayi/food-recipe-finder-models"
