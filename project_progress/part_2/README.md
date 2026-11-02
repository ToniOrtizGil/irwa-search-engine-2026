# Part 2 - Indexing and Evaluation

Tag: `IRWA-2026-part-2` · The report (PDF) is submitted in Aula Global.

All commands are run **from the project root** with the virtual environment active
(installation steps are in the root [README](../../README.md#quick-start-part-1)).

## 1. Input data

| File | Content |
|---|---|
| `data/fashion_products_dataset.json` | corpus (provided by the instructors, git-ignored) |
| `data/validation_labels.csv` | relevance judgments for 2 queries (provided by the instructors, git-ignored) |
| `data/processed/processed_corpus.pkl` | processed corpus of Part 1 (`python -m project_progress.part_1.run_preprocessing`); the notebook creates it if it does not exist |
| `project_progress/part_2/own_queries_labels.csv` | our relevance judgments: top 20 documents of each of our 5 queries (`query_id, query, rank, pid, title, labels`) |

## 2. Run

```bash
# executes every cell (~30 s)
jupyter nbconvert --to notebook --execute --inplace project_progress/part_2/part2_indexing_evaluation.ipynb
# or open it with: jupyter notebook project_progress/part_2/part2_indexing_evaluation.ipynb and Run All
```

## 3. Code

| File | Content |
|---|---|
| `myapp/search/algorithms.py` | `create_index_tfidf()` (inverted index, idf, tf-idf weights and document lengths), `conjunctive_search()` (AND queries), `rank_documents()` (cosine similarity of tf-idf vectors, log base 2) and `search_tf_idf()` (query -> ranked results) |
| `myapp/evaluation/metrics.py` | `precision_at_k`, `recall_at_k`, `avg_precision_at_k`, `f1_at_k`, `rr_at_k`, `ndcg_at_k`, `evaluate_query` and `mean_metrics` (MAP, MRR) |
| `project_progress/part_2/part2_indexing_evaluation.ipynb` | builds the index, runs our 5 test queries and computes all the metrics |

Using the search engine from Python:

```python
import pickle
from myapp.search.algorithms import create_index_tfidf, search_tf_idf

corpus = pickle.load(open("data/processed/processed_corpus.pkl", "rb"))
index, weights, idf, doc_length = create_index_tfidf(corpus)
search_tf_idf("men slim jeans blue", index, weights, idf, doc_length)[:10]   # list of (pid, score)
```

## 4. Notebook sections

1. Packages
2. Load the processed corpus of Part 1
3. Build the inverted index
4. Conjunctive (AND) queries
5. Propose five test queries (based on the most frequent title terms)
6. Rank the results with TF-IDF and cosine similarity (checked with the examples of class)
7. Evaluation metrics (checked with the examples of the slides)
8. Evaluation with `validation_labels.csv` (K = 10)
9. Our test queries: relevance judgments and evaluation (K = 10)
10. Some problems of the system (used in the analysis of the report)
