"""Processes the whole corpus and saves it for Part 2.

Run from the root: python -m project_progress.part_1.run_preprocessing
"""

import time

from myapp.preprocessing.fields import (build_processed_corpus, load_raw_corpus,
                                        numeric_frame, save_processed_corpus)
from myapp.preprocessing.text_processing import build_terms

DATA_PATH = "data/fashion_products_dataset.json"
OUTPUT_PATH = "data/processed/processed_corpus.pkl"

start_time = time.time()
df = load_raw_corpus(DATA_PATH)
print("Loaded {} documents from {}".format(len(df), DATA_PATH))

corpus = build_processed_corpus(df)
print("Processed {} documents in {} seconds".format(len(corpus), round(time.time() - start_time, 2)))

for doc in list(corpus.values())[:2]:
    print("-" * 80)
    print("pid:", doc["pid"])
    print("title:", doc["meta"]["title"])
    print("title terms:", doc["tokens"]["title"])
    print("product_details terms:", doc["tokens"]["product_details"])
    print("price / discount / rating:", doc["meta"]["selling_price"],
          doc["meta"]["discount"], doc["meta"]["average_rating"])

print("-" * 80)
for query in ["women full sleeve sweatshirt cotton", "men slim jeans blue"]:
    print("query '{}' -> {}".format(query, build_terms(query)))

print("\nNumeric fields:")
print(numeric_frame(corpus).describe().round(2))

save_processed_corpus(corpus, OUTPUT_PATH)
print("\nSaved processed corpus to", OUTPUT_PATH)
