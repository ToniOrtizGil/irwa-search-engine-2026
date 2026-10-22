"""Text pre-processing (Part 1): same build_terms() for documents and queries, as in the lab."""

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
    """Lowercase and split on spaces and punctuation.

    Also joins "t-shirt" into "tshirt" and normalizes gender words (woman -> women).
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"\bt[\s\-]?shirt", "tshirt", text)  # avoid splitting into "t" + "shirt"
    text = re.sub(r"[^a-z0-9]", " ", text)             # also splits "look.Designed"
    tokens = text.split()
    return [GENDER_WORDS.get(t, t) for t in tokens]


def remove_numbers(tokens):
    """Remove tokens that are only digits (numeric info is kept in separate fields)."""
    return [t for t in tokens if not t.isdigit()]


def remove_stopwords(tokens):
    """Remove English stop words (nltk list)."""
    return [t for t in tokens if t not in stop_words]


def remove_short_tokens(tokens):
    """Remove one-character tokens, e.g. the "s" left by "women's"."""
    return [t for t in tokens if len(t) > 1]


def stem_tokens(tokens):
    """Apply the Porter stemmer to every token."""
    return [stemmer.stem(t) for t in tokens]


def build_terms(line, stem=True):
    """Return the index terms of a text. Used for documents and queries.

    stem=False skips stemming (used for the word clouds).
    """
    line = tokenize(line)
    line = remove_numbers(line)
    line = remove_stopwords(line)
    line = remove_short_tokens(line)
    if stem:
        line = stem_tokens(line)
    return line
