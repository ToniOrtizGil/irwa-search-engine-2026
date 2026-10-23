# Part 1 — Text Processing and Exploratory Data Analysis

Tag: `IRWA-2026-part-1` · Report: [`IRWA-2026-u233542-u232701-u232752-part-1.pdf`](IRWA-2026-u233542-u232701-u232752-part-1.pdf) (also submitted in Aula Global).

All commands are run **from the project root** with the virtual environment active
(installation steps are in the root [README](../../README.md#quick-start-part-1)).

## 1. Input data

Place the files provided by the instructors in `data/` (the folder is git-ignored):

| File | Content |
|---|---|
| `data/fashion_products_dataset.json` | 28,080 products (list of JSON objects) |
| `data/validation_labels.csv` | `title, pid, query_id, labels` — 40 judged documents for 2 queries |

## 2. Pre-processing

```bash
python -m project_progress.part_1.run_preprocessing
```

It reads `data/fashion_products_dataset.json` and saves the processed corpus in
`data/processed/processed_corpus.pkl` (the paths are the constants `DATA_PATH` and `OUTPUT_PATH` at the
top of the script).

The output is a dict `pid -> document` where each document has:

* `meta`: the fields that every search result must return (`pid, title, description, brand,
  category, sub_category, product_details, seller, out_of_stock, selling_price, discount,
  actual_price, average_rating, url`), with numeric fields already parsed to `float`/`bool`
  and `product_details` flattened to a dict;
* `tokens`: one processed term list per textual field (`title, description, brand, category,
  sub_category, product_details, seller`), to allow field-weighted indexing in Part 2;
* `search_tokens`: the combined field (`title + description + brand + sub_category +
  product_details values`) used as default index field.

Using the functions from Python:

```python
from myapp.preprocessing.text_processing import build_terms
import pickle

build_terms("men slim jeans blue")               # ['men', 'slim', 'jean', 'blue']
build_terms("Rich look.Designed for comfort", stem=False)
corpus = pickle.load(open("data/processed/processed_corpus.pkl", "rb"))
```

`build_terms(line, stem=True)` is used for documents and queries. Every step is also available as
a separate function (`tokenize`, `remove_numbers`, `remove_stopwords`, `remove_short_tokens`,
`stem_tokens`).

## 3. Exploratory data analysis

```bash
# executes every cell and regenerates figures/ (~3 min)
# (or open it with: jupyter notebook project_progress/part_1/part1_eda.ipynb and Run All)
jupyter nbconvert --to notebook --execute --inplace project_progress/part_1/part1_eda.ipynb
```

The notebook also runs the pre-processing, so it reproduces every number and figure of the report.
Sections:

1. Packages
2. Load data (missing values and duplicates)
3. Pre-processing of title and description: effect of each step on |V| and tokens
4. Process the whole corpus and save it
5. category / sub_category / brand / seller (df, idf) and product_details attributes
6. Numeric fields: conversion and checks
7. Validation labels: in which fields do the query terms appear
8. Word counting distribution and average sentence length
9. Vocabulary size, Zipf's law and Heaps' law
10. Most frequent words and word clouds
11. Ranking of products by rating, price and discount
12. out_of_stock distribution
13. Top sellers and brands
14. Entity recognition (spaCy `en_core_web_sm`)
15. Other checks (gender in URL vs title)

## 4. Outputs

| Path | Produced by |
|---|---|
| `data/processed/processed_corpus.pkl` | `run_preprocessing.py` and the notebook |
| `project_progress/part_1/figures/*.png` | notebook |
