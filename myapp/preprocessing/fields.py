"""Field handling: numeric fields, product_details and the processed corpus."""

import json
import os
import pickle

import pandas as pd

from myapp.preprocessing.text_processing import build_terms

TEXT_FIELDS = ["title", "description", "brand", "category", "sub_category",
               "product_details", "seller"]

# category is almost constant and seller describes the shop, so they stay out of the combined field
SEARCH_FIELDS = ["title", "description", "brand", "sub_category", "product_details"]

# codes and constant values, useless as terms
EXCLUDED_DETAIL_KEYS = ["style code", "model name", "model number", "series",
                        "net quantity", "number of contents in sales package",
                        "pack of", "country of origin", "fabric care", "size"]

# "Hooded: Yes" says nothing without the key
EXCLUDED_DETAIL_VALUES = ["yes", "no", "na", "n/a", "none", "-"]


def load_raw_corpus(path):
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    return pd.DataFrame(data)


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def parse_price(value):
    if is_missing(value):
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def parse_discount(value):
    if is_missing(value):
        return None
    number = str(value).replace("% off", "").replace("%", "").replace(",", "").strip()
    try:
        return float(number)
    except ValueError:
        return None


def parse_rating(value):
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
    result = {}
    if not isinstance(details, list):
        return result
    for item in details:
        for key, value in item.items():
            result[key] = str(value)
    return result


def product_details_to_text(details):
    """Only the values are indexed, the keys ("Fabric", "Color"...) are in almost every product."""
    values = []
    for key, value in details.items():
        if key.strip().lower() in EXCLUDED_DETAIL_KEYS:
            continue
        if value.strip().lower() in EXCLUDED_DETAIL_VALUES:
            continue
        values.append(value)
    return " ".join(values)


def build_processed_document(raw):
    """Returns pid, meta (output fields), tokens (terms per field) and search_tokens (combined field)."""
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
    corpus = {}
    for raw in df.to_dict(orient="records"):
        doc = build_processed_document(raw)
        corpus[doc["pid"]] = doc
    return corpus


def numeric_frame(corpus):
    rows = []
    for doc in corpus.values():
        m = doc["meta"]
        rows.append([m["pid"], m["out_of_stock"], m["selling_price"], m["discount"],
                     m["actual_price"], m["average_rating"]])
    columns = ["pid", "out_of_stock", "selling_price", "discount", "actual_price", "average_rating"]
    return pd.DataFrame(rows, columns=columns).set_index("pid")


def save_processed_corpus(corpus, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fp:
        pickle.dump(corpus, fp)
