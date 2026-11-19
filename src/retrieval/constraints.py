"""Extract time and diet constraints from a free-text query with regular expressions.

Returns the query with the constraint phrases removed (so they do not pollute the
text match) and the structured constraints used to filter / re-rank recipes.
"""
import re
from dataclasses import dataclass, field

_UNIT = r"(?P<unit>minutes?|mins?|hours?|hrs?)"
_NUM = r"(?P<num>\d+(?:\.\d+)?)"

# (pattern, strict). strict=True means minutes < limit, otherwise minutes <= limit.
TIME_PATTERNS = [
    (re.compile(rf"\b(?:under|less than|below|fewer than)\s+{_NUM}\s*{_UNIT}\b"), True),
    (re.compile(rf"\b(?:within|in|no more than|at most|up to)\s+{_NUM}\s*{_UNIT}"
                rf"(?:\s+or\s+(?:less|under|fewer))?\b"), False),
    (re.compile(rf"\b{_NUM}\s*{_UNIT}\s+or\s+(?:less|under|fewer)\b"), False),
]
_HOUR_WORD = [  # "under an hour", "within an hour"
    (re.compile(r"\b(?:under|less than|below)\s+an?\s+hour\b"), True),
    (re.compile(r"\b(?:within|in)\s+an?\s+hour\b"), False),
]

DIET_PATTERNS = {
    "vegetarian": re.compile(r"\bvegetarian\b"),
    "vegan": re.compile(r"\bvegan\b"),
    "gluten-free": re.compile(r"\bgluten[- ]?free\b"),
    "dairy-free": re.compile(r"\bdairy[- ]?free\b"),
    "low-carb": re.compile(r"\blow[- ]?carb(?:ohydrate)?s?\b"),
    "low-fat": re.compile(r"\blow[- ]?fat\b"),
    "low-sodium": re.compile(r"\blow[- ]?sodium\b"),
    "low-calorie": re.compile(r"\blow[- ]?cal(?:orie)?s?\b"),
}


@dataclass
class Constraints:
    max_minutes: float = None
    strict: bool = False
    diets: list = field(default_factory=list)
    text: str = ""          # query with constraint phrases removed

    def satisfied_by(self, minutes, diet_tags):
        if self.max_minutes is not None:
            if (minutes >= self.max_minutes) if self.strict else (minutes > self.max_minutes):
                return False
        tags = set(diet_tags.split("|")) if diet_tags else set()
        return all(d in tags for d in self.diets)

    @property
    def any(self):
        return self.max_minutes is not None or bool(self.diets)


def extract_constraints(query):
    q = " ".join(str(query).lower().split())
    c = Constraints()
    for pat, strict in TIME_PATTERNS:
        m = pat.search(q)
        if m:
            val = float(m.group("num"))
            if m.group("unit").startswith(("h",)):
                val *= 60
            c.max_minutes, c.strict = val, strict
            q = (q[:m.start()] + " " + q[m.end():])
            break
    else:
        for pat, strict in _HOUR_WORD:
            m = pat.search(q)
            if m:
                c.max_minutes, c.strict = 60.0, strict
                q = (q[:m.start()] + " " + q[m.end():])
                break
    for diet, pat in DIET_PATTERNS.items():
        m = pat.search(q)
        if m:
            c.diets.append(diet)
            q = (q[:m.start()] + " " + q[m.end():])
    c.text = " ".join(q.split())
    return c
