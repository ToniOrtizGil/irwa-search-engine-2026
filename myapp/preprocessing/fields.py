"""Field pre-processing (Part 1): numeric fields, product_details and the processed corpus."""

import json
import os
import pickle

import pandas as pd

from myapp.preprocessing.text_processing import build_terms

TEXT_FIELDS = ["title", "description", "brand", "category", "sub_category",
               "product_details", "seller"]

# category (96.6% same value, idf ~ 0) and seller (describes the shop) are left out
SEARCH_FIELDS = ["title", "description", "brand", "sub_category", "product_details"]

# codes and constant values that are useless as terms
EXCLUDED_DETAIL_KEYS = ["style code", "model name", "model number", "series",
                        "net quantity", "number of contents in sales package",
                        "pack of", "country of origin", "fabric care", "size"]

# values that mean nothing without their key ("Hooded: Yes")
EXCLUDED_DETAIL_VALUES = ["yes", "no", "na", "n/a", "none", "-"]


def load_raw_corpus(path):
    """Load the json corpus into a DataFrame (one row per product)."""
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    return pd.DataFrame(data)


def is_missing(value):
    """Return True if the value is None, NaN or an empty string."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def parse_price(value):
    """Convert a price like "2,999" to 2999.0 (None if missing)."""
    if is_missing(value):
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def parse_discount(value):
    """Convert a discount like "69% off" to 69.0 (None if missing)."""
    if is_missing(value):
        return None
    number = str(value).replace("% off", "").replace("%", "").replace(",", "").strip()
    try:
        return float(number)
    except ValueError:
        return None


def parse_rating(value):
    """Convert the rating to a float; only values between 0 and 5 are valid."""
    if is_missing(value):
        return None
    try:
        rating = float(value)
    except ValueError:
        return None
    if 0 <= rating <= 5:
        return rating
    return None


def flatten_product_details(details):
    """Turn [{"Fabric": "Cotton"}, {"Color": "Blue"}] into {"Fabric": "Cotton", "Color": "Blue"}."""
    result = {}
    if not isinstance(details, list):
        return result
    for item in details:
        for key, value in item.items():
            result[key] = str(value)
    return result


def product_details_to_text(details):
    """Join the useful values of product_details into one text.

    Only values are indexed: keys like "Fabric" appear in almost every product.
    """
    values = []
    for key, value in details.items():
        if key.strip().lower() in EXCLUDED_DETAIL_KEYS:
            continue
        if value.strip().lower() in EXCLUDED_DETAIL_VALUES:
            continue
        values.append(value)
    return " ".join(values)


def build_processed_document(raw):
    """Process one product.

    Returns a dict with pid, meta (output fields), tokens (terms of each field)
    and search_tokens (combined field).
    """
    details = flatten_product_details(raw.get("product_details"))

    meta = {
        "pid": raw.get("pid"),
        "title": raw.get("title") or "",
        "description": raw.get("description") or "",
        "brand": raw.get("brand") or "",
        "category": raw.get("category") or "",
        "sub_category": raw.get("sub_category") or "",
        "product_details": details,
        "seller": raw.get("seller") or "",
        "out_of_stock": bool(raw.get("out_of_stock", False)),
        "selling_price": parse_price(raw.get("selling_price")),
        "discount": parse_discount(raw.get("discount")),
        "actual_price": parse_price(raw.get("actual_price")),
        "average_rating": parse_rating(raw.get("average_rating")),
        "url": raw.get("url") or "",
    }

    tokens = {}
    for field in TEXT_FIELDS:
        if field == "product_details":
            tokens[field] = build_terms(product_details_to_text(details))
        else:
            tokens[field] = build_terms(meta[field])

    search_tokens = []
    for field in SEARCH_FIELDS:
        search_tokens += tokens[field]

    return {"pid": meta["pid"], "meta": meta, "tokens": tokens, "search_tokens": search_tokens}


def build_processed_corpus(df):
    """Process all the products and return a dict pid -> processed document."""
    corpus = {}
    for raw in df.to_dict(orient="records"):
        doc = build_processed_document(raw)
        corpus[doc["pid"]] = doc
    return corpus


def numeric_frame(corpus):
    """Return a DataFrame (index pid) with the numeric fields, for filtering and ranking in Part 3."""
    rows = []
    for doc in corpus.values():
        m = doc["meta"]
        rows.append([m["pid"], m["out_of_stock"], m["selling_price"], m["discount"],
                     m["actual_price"], m["average_rating"]])
    columns = ["pid", "out_of_stock", "selling_price", "discount", "actual_price", "average_rating"]
    return pd.DataFrame(rows, columns=columns).set_index("pid")


def save_processed_corpus(corpus, path):
    """Save the processed corpus with pickle to reuse it in the next parts."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fp:
        pickle.dump(corpus, fp)
