"""Build the recipe corpus (>=200 recipes covering the 10 dishes) from Food.com RAW_recipes.csv."""
import ast
import string

import numpy as np
import pandas as pd

from src.config import SEED

# (include regex, exclude regex) matched against the lower-cased recipe name.
# Excludes remove components / variants (sauce, dough, casserole, veggie burgers ...).
DISH_PATTERNS = {
    "pizza": (r"\bpizzas?\b",
              r"\b(?:dough|crust|sauce|seasoning|rolls?|bread|dip|casserole|pasta|spaghetti|soup|salad|"
              r"bites|poppers|sandwich(?:es)?|burgers?|bagels?|quesadillas?|tarts?|cupcakes?|cookies?)\b"),
    "sushi": (r"\bsushi\b",
              r"\b(?:sauce|vinegar|seasoning|salad|casserole|bake|pizza|burritos?|dressing)\b"),
    "hamburger": (r"\b(?:hamburgers?|cheeseburgers?|burgers?)\b",
                  r"\b(?:veggie|vegetarian|vegan|turkey|chicken|salmon|tuna|bean|beans|lentil|portobello|"
                  r"sauce|casserole|soup|salad|dip|pie|seasoning|bake|helper|stroganoff|sliders?)\b"),
    "ice_cream": (r"\bice[- ]?cream\b",
                  r"\b(?:cake|pie|sandwich(?:es)?|bars?|cookies?|cones?|topping|sauce|float|soda|punch|"
                  r"brownies?|cups?|rolls?|bombe|truffles?|cupcakes?|tart|waffles?|muffins?|bread|fudge|"
                  r"candy|dip|shakes?|milkshakes?|smoothies?|pizza)\b"),
    "caesar_salad": (r"\bcaesar\b.*\bsalad\b|\bsalad\b.*\bcaesar\b",
                     r"\b(?:dressing|sandwich(?:es)?|wraps?|pasta|pizza|croutons?|soup|dip|bites|pita|panini|burgers?)\b"),
    "spaghetti_carbonara": (r"\bcarbonara\b",
                            r"\b(?:sauce|casserole|bake|baked|soup|salad|pizza|pie|stuffed|burgers?|risotto|"
                            r"sandwich(?:es)?|quiche)\b"),
    "pancakes": (r"\bpancakes?\b",
                 r"\b(?:mix|syrup|topping|sauce|casserole|bake|baked|batter|cakes?|muffins?|sandwich(?:es)?|"
                 r"bites|dippers?|sausage|corn dogs?|waffles?|dutch|puppies|pizza|tacos?|rolls?|pie|bars?|cookies?)\b"),
    "omelette": (r"\bomel(?:et|ette)s?\b",
                 r"\b(?:casserole|sandwich(?:es)?|muffins?|pie|bake|baked|cups?|rolls?|pizza|wraps?|cake|sauce|"
                 r"bites|burritos?|quesadillas?|tacos?)\b"),
    "french_fries": (r"\bfrench fries\b|\bfries\b",
                     r"\b(?:sauce|seasoning|dip|soup|casserole|salad|burgers?|sandwich(?:es)?|pie|pizza|tacos?|"
                     r"stuffed|bake|chili|poutine|nachos|dogs?|dressing|chicken|fish|fried rice|bread)\b"),
    "chicken_curry": (r"\bchicken\b.*\bcurr(?:y|ied)\b|\bcurr(?:y|ied)\b.*\bchicken\b",
                      r"\b(?:soup|salad|sandwich(?:es)?|wraps?|pie|pizza|paste|powder|sauce|casserole|dip|"
                      r"burgers?|spread|tacos?|noodles?|pasta|pot pie|sliders?|quesadillas?|bites|meatballs?|"
                      r"kabobs?|skewers?)\b"),
}

# Food.com tags that we treat as dietary attributes (only the ones present in the data are kept).
DIET_TAGS = ["vegetarian", "vegan", "gluten-free", "dairy-free", "low-carb", "low-fat",
             "low-sodium", "low-calorie"]

MIN_MINUTES, MAX_MINUTES = 5, 480
COLUMNS = ["recipe_id", "title", "dish", "minutes", "diet_tags", "ingredients", "steps", "description"]


def _parse_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        return list(ast.literal_eval(x))
    except Exception:
        return []


def build_corpus(raw, per_class=30, seed=SEED):
    """raw = pandas DataFrame read from RAW_recipes.csv. Returns the corpus DataFrame."""
    df = raw.copy()
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df = df[(df["name"] != "")
            & df["minutes"].between(MIN_MINUTES, MAX_MINUTES)
            & (df["n_ingredients"] >= 3) & (df["n_steps"] >= 2)]
    df["name_l"] = df["name"].str.lower()
    rng = np.random.RandomState(seed)

    parts = []
    for dish, (inc, exc) in DISH_PATTERNS.items():
        m = df[df["name_l"].str.contains(inc, regex=True) & ~df["name_l"].str.contains(exc, regex=True)]
        m = m.drop_duplicates(subset="name_l")
        m = m.iloc[rng.permutation(len(m))[:per_class]].copy()
        m["dish"] = dish
        parts.append(m)
    out = pd.concat(parts, ignore_index=True)

    tags = out["tags"].apply(_parse_list)
    out["diet_tags"] = tags.apply(lambda t: "|".join(sorted(set(t) & set(DIET_TAGS))))
    out["ingredients"] = out["ingredients"].apply(lambda x: "; ".join(_parse_list(x)))
    out["steps"] = out["steps"].apply(lambda x: " || ".join(_parse_list(x)))
    out["description"] = out["description"].fillna("").astype(str)
    out["title"] = out["name"].apply(string.capwords)
    out = out.rename(columns={"id": "recipe_id"})
    return out[COLUMNS].reset_index(drop=True)
