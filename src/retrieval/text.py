"""Text normalisation, tokenisation, stop-word removal, stemming / lemmatisation."""
import re
import unicodedata
from functools import lru_cache

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN = re.compile(r"[a-z0-9]+")


def normalize(text):
    """Lower-case, strip accents (jalapeño -> jalapeno), keep letters/digits only."""
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.replace("'", "")


@lru_cache(maxsize=None)
def _stemmer():
    from nltk.stem import SnowballStemmer
    return SnowballStemmer("english")


@lru_cache(maxsize=None)
def _lemmatizer():
    import nltk
    from nltk.stem import WordNetLemmatizer
    try:
        WordNetLemmatizer().lemmatize("tomatoes")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    return WordNetLemmatizer()


@lru_cache(maxsize=None)
def _stem(tok):
    return _stemmer().stem(tok)


@lru_cache(maxsize=None)
def _lemma(tok):
    wnl = _lemmatizer()
    return wnl.lemmatize(wnl.lemmatize(tok, "n"), "v")  # noun form first, then verb form


class TextProcessor:
    """mode: 'raw'   = lower-case + tokenise only
             'none'  = + normalisation (accents) + stop-word removal
             'stem'  = 'none' + Snowball stemming
             'lemma' = 'none' + WordNet lemmatisation
    """

    def __init__(self, mode="stem"):
        assert mode in ("raw", "none", "stem", "lemma")
        self.mode = mode

    def __call__(self, text):
        if self.mode == "raw":
            return _TOKEN.findall(str(text).lower())
        toks = [t for t in _TOKEN.findall(normalize(text)) if t not in ENGLISH_STOP_WORDS]
        if self.mode == "stem":
            return [_stem(t) for t in toks]
        if self.mode == "lemma":
            return [_lemma(t) for t in toks]
        return toks
