"""Central configuration. Every script/notebook imports from here."""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# 10-class subset of Food-101
CLASSES = [
    "pizza", "sushi", "hamburger", "ice_cream", "caesar_salad",
    "spaghetti_carbonara", "pancakes", "omelette", "french_fries", "chicken_curry",
]

IMG_SIZE = 224
