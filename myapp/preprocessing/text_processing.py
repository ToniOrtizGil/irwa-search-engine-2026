"""Text pre-processing: the same build_terms() is used for documents and queries."""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

GENDER_WORDS = {"woman": "women", "womens": "women", "womans": "women",
                "man": "men", "mens": "men", "mans": "men"}


def tokenize(text):
    """Lowercase, split on anything that is not a letter or digit and unify gender words."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    # otherwise "t-shirt" becomes "t" + "shirt" and matches normal shirts
    text = re.sub(r"\bt[\s\-]?shirt", "tshirt", text)
    text = re.sub(r"[^a-z0-9]", " ", text)
    return [GENDER_WORDS.get(t, t) for t in text.split()]


def remove_numbers(tokens):
    return [t for t in tokens if not t.isdigit()]


def remove_stopwords(tokens):
    return [t for t in tokens if t not in stop_words]


def remove_short_tokens(tokens):
    # drops the "s" left by "women's"
    return [t for t in tokens if len(t) > 1]


def stem_tokens(tokens):
    return [stemmer.stem(t) for t in tokens]


def build_terms(line, stem=True):
    """Index terms of a text. stem=False is only used for the word clouds."""
    line = tokenize(line)
    line = remove_numbers(line)
    line = remove_stopwords(line)
    line = remove_short_tokens(line)
    if stem:
        line = stem_tokens(line)
    return line
