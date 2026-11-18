"""Generate the labelled test queries (ground truth) from the recipe corpus.

Relevance is defined by explicit, documented criteria on recipe metadata / ingredient lists
(see the `criteria` column), never by looking at any retrieval model's output.

Query types
  time      dish + time limit                  e.g. "quick pizza under 30 minutes"
  diet      diet + dish                        e.g. "vegetarian pancakes"
  ingredient dish + ingredient                 e.g. "chicken curry with coconut milk"
  combo     diet + dish + time limit           e.g. "vegan french fries under 45 minutes"
  semantic  paraphrase WITHOUT the dish name   e.g. "raw fish and seasoned rice rolled in seaweed"
            (relevant = every recipe of that dish)
"""
import re

import pandas as pd

from src.config import CLASSES, DISH_NAMES

TIME_TEMPLATES = [  # (template, strict)  strict -> minutes < x, else minutes <= x
    ("quick {dish} under {x} minutes", True),
    ("{dish} in {x} minutes or less", False),
    ("{dish} that takes less than {x} min", True),
    ("easy {dish} within {x} minutes", False),
]
TIME_LIMITS = [20, 30, 45, 60]

DIET_PHRASE = {"vegetarian": ["vegetarian"], "vegan": ["vegan"], "gluten-free": ["gluten free", "gluten-free"],
               "dairy-free": ["dairy free"], "low-carb": ["low carb"], "low-fat": ["low fat"]}
DIET_CANDIDATES = [("pancakes", "vegetarian"), ("pizza", "vegetarian"), ("omelette", "vegetarian"),
                   ("french_fries", "vegan"), ("pancakes", "vegan"), ("ice_cream", "vegetarian"),
                   ("chicken_curry", "gluten-free"), ("pizza", "gluten-free"), ("sushi", "gluten-free"),
                   ("hamburger", "low-carb"), ("caesar_salad", "low-carb"), ("omelette", "low-carb"),
                   ("spaghetti_carbonara", "low-fat"), ("pancakes", "low-fat"), ("french_fries", "gluten-free"),
                   ("omelette", "gluten-free"), ("chicken_curry", "dairy-free"), ("pancakes", "dairy-free")]

ING_CANDIDATES = {
    "pizza": [("mushrooms", r"\bmushrooms?\b"), ("pepperoni", r"\bpepperoni\b"), ("spinach", r"\bspinach\b")],
    "sushi": [("avocado", r"\bavocados?\b"), ("cucumber", r"\bcucumbers?\b"), ("salmon", r"\bsalmon\b"),
              ("crab", r"\bcrab")],
    "hamburger": [("bacon", r"\bbacon\b"), ("onions", r"\bonions?\b"), ("cheese", r"\bcheese\b")],
    "ice_cream": [("strawberries", r"\bstrawberr(?:y|ies)\b"), ("chocolate", r"\bchocolate\b"),
                  ("peanut butter", r"\bpeanut butter\b"), ("mint", r"\bmint\b")],
    "caesar_salad": [("chicken", r"\bchicken\b"), ("bacon", r"\bbacon\b"), ("parmesan", r"\bparmesan\b"),
                     ("anchovies", r"\banchov")],
    "spaghetti_carbonara": [("bacon", r"\bbacon\b"), ("pancetta", r"\bpancetta\b"),
                            ("parmesan", r"\bparmesan\b"), ("peas", r"\bpeas?\b")],
    "pancakes": [("blueberries", r"\bblueberr(?:y|ies)\b"), ("bananas", r"\bbananas?\b"),
                 ("chocolate chips", r"\bchocolate chips?\b"), ("apples", r"\bapples?\b")],
    "omelette": [("spinach", r"\bspinach\b"), ("mushrooms", r"\bmushrooms?\b"), ("ham", r"\bham\b"),
                 ("cheese", r"\bcheese\b")],
    "french_fries": [("garlic", r"\bgarlic\b"), ("parmesan", r"\bparmesan\b"),
                     ("sweet potatoes", r"\bsweet potato(?:es)?\b"), ("rosemary", r"\brosemary\b")],
    "chicken_curry": [("coconut milk", r"\bcoconut milk\b"), ("potatoes", r"\bpotato(?:es)?\b"),
                      ("yogurt", r"\byogh?urt\b"), ("spinach", r"\bspinach\b")],
}

COMBO_CANDIDATES = [("pizza", "vegetarian", 45), ("pancakes", "vegetarian", 30), ("omelette", "vegetarian", 20),
                    ("french_fries", "vegan", 45), ("chicken_curry", "gluten-free", 60),
                    ("sushi", "gluten-free", 45), ("pancakes", "vegan", 30), ("pizza", "vegetarian", 60)]

SEMANTIC = [
    ("pizza", "baked flatbread topped with tomato sauce and melted mozzarella"),
    ("sushi", "raw fish and seasoned rice rolled in seaweed"),
    ("hamburger", "grilled beef patty served in a toasted bun"),
    ("ice_cream", "cold creamy frozen dessert to cool down on a hot day"),
    ("caesar_salad", "crisp romaine lettuce with croutons and a garlicky anchovy dressing"),
    ("spaghetti_carbonara", "italian pasta with eggs, cheese and crispy cured pork"),
    ("pancakes", "fluffy breakfast stack served with maple syrup"),
    ("omelette", "folded egg breakfast filled with vegetables and cheese"),
    ("french_fries", "crispy golden potato sticks fried until crunchy"),
    ("chicken_curry", "spiced chicken simmered in a rich sauce, served with rice"),
]


def _select(df, dish=None, max_min=None, strict=False, diet=None, ing=None):
    m = pd.Series(True, index=df.index)
    if dish is not None:
        m &= df["dish"] == dish
    if max_min is not None:
        m &= (df["minutes"] < max_min) if strict else (df["minutes"] <= max_min)
    if diet is not None:
        m &= df["diet_tags"].str.split("|").apply(lambda t: diet in t)
    if ing is not None:
        m &= df["ingredients"].str.lower().str.contains(ing, regex=True)
    return df[m]


def build_queries(corpus, n_time=8, n_diet=8, n_combo=4):
    n_dish = corpus["dish"].value_counts().to_dict()
    rows = []

    def add(qtype, text, rel, criteria):
        rows.append({"type": qtype, "query": text, "relevant_ids": "|".join(map(str, rel["recipe_id"])),
                     "n_relevant": len(rel), "criteria": criteria})

    # 1) dish + time limit
    k = 0
    for dish in CLASSES:
        if k >= n_time or dish not in n_dish:
            continue
        tmpl, strict = TIME_TEMPLATES[k % len(TIME_TEMPLATES)]
        r = k % len(TIME_LIMITS)  # rotate the starting limit so different queries use different limits
        for x in TIME_LIMITS[r:] + TIME_LIMITS[:r]:
            rel = _select(corpus, dish=dish, max_min=x, strict=strict)
            if 2 <= len(rel) <= 0.7 * n_dish[dish]:
                add("time", tmpl.format(dish=DISH_NAMES[dish], x=x), rel,
                    f"dish={dish} & minutes{'<' if strict else '<='}{x}")
                k += 1
                break

    # 2) diet + dish
    k = 0
    for dish, diet in DIET_CANDIDATES:
        if k >= n_diet:
            break
        if dish not in n_dish:
            continue
        rel = _select(corpus, dish=dish, diet=diet)
        if 2 <= len(rel) <= 0.8 * n_dish[dish]:
            phrase = DIET_PHRASE[diet][k % len(DIET_PHRASE[diet])]
            add("diet", f"{phrase} {DISH_NAMES[dish]}", rel, f"dish={dish} & diet_tag={diet}")
            k += 1

    # 3) dish + ingredient (one per dish, first valid candidate)
    for dish in CLASSES:
        if dish not in n_dish:
            continue
        for text, rx in ING_CANDIDATES[dish]:
            rel = _select(corpus, dish=dish, ing=rx)
            if 2 <= len(rel) <= 0.8 * n_dish[dish]:
                add("ingredient", f"{DISH_NAMES[dish]} with {text}", rel,
                    f"dish={dish} & ingredients match /{rx}/")
                break

    # 4) semantic paraphrases (no dish name in the query)
    for dish, text in SEMANTIC:
        if dish in n_dish:
            add("semantic", text, _select(corpus, dish=dish), f"dish={dish}")

    # 5) diet + dish + time limit
    k = 0
    for dish, diet, x in COMBO_CANDIDATES:
        if k >= n_combo:
            break
        if dish not in n_dish:
            continue
        rel = _select(corpus, dish=dish, diet=diet, max_min=x, strict=True)
        if 1 <= len(rel) <= 15:
            add("combo", f"{DIET_PHRASE[diet][0]} {DISH_NAMES[dish]} under {x} minutes", rel,
                f"dish={dish} & diet_tag={diet} & minutes<{x}")
            k += 1

    q = pd.DataFrame(rows)
    q.insert(0, "query_id", [f"q{i + 1:02d}" for i in range(len(q))])
    return q
