# Part 3 - Ranking & Filtering

Tag: `IRWA-2026-part-3` · The report (PDF) is submitted in Aula Global.

All commands are run **from the project root** with the virtual environment active
(installation steps are in the root [README](../../README.md#quick-start-part-1); Part 3 also needs `gensim`,
which is in `requirements.txt`).

## 1. Input data

| File | Content |
|---|---|
| `data/fashion_products_dataset.json` | corpus (provided by the instructors, git-ignored) |
| `data/validation_labels.csv` | relevance judgments for 2 queries (provided by the instructors, git-ignored) |
| `data/processed/processed_corpus.pkl` | processed corpus of Part 1 (`python -m project_progress.part_1.run_preprocessing`); the notebook creates it if it does not exist |

## 2. Run

```bash
# executes every cell (~35 s)
jupyter nbconvert --to notebook --execute --inplace project_progress/part_3/part3_ranking.ipynb
# or open it with: jupyter notebook project_progress/part_3/part3_ranking.ipynb and Run All
```

The notebook also writes `project_progress/part_3/word2vec_top20.csv` (top-20 documents of our 5 test queries
with word2vec + cosine: `query_id, query, rank, pid, title, score`).

## 3. Selecting the ranking method

Every method uses the same pre-processing (`build_terms`) and the same conjunctive (AND) search; only the order
of the results changes.

| `method` | Ranking | Function |
|---|---|---|
| `"tfidf"` | TF-IDF + cosine similarity (Part 2, log base 2) | `rank_documents` in `myapp/search/algorithms.py` |
| `"bm25"` | BM25 of the theory, log base 2, `k1 = 1.2`, `b = 0.75` | `rank_bm25` in `myapp/search/ranking.py` |
| `"ours"` | BM25 x title factor x stock factor x value factor (rating and price percentiles among the results) | `rank_our_score` in `myapp/search/ranking.py` |
| `"word2vec"` | cosine between the average word2vec vectors of the query and the document | `rank_word2vec` in `myapp/search/ranking.py` |

```python
import pickle
from myapp.search.ranking import build_search_engine, search

corpus = pickle.load(open("data/processed/processed_corpus.pkl", "rb"))
engine = build_search_engine(corpus)          # index + BM25 statistics + word2vec (~25 s)
search("men white sneakers", engine, method="bm25", top_k=10)   # list of (pid, score)
search("men white sneakers", engine, method="ours", top_k=10)
search("men white sneakers", engine, method="word2vec", top_k=20)
```

`build_search_engine(corpus, with_word2vec=False)` skips word2vec if only `tfidf`, `bm25` or `ours` are needed.
The BM25 parameters can be changed with `rank_bm25(..., k1=..., b=...)`, and the word2vec parameters with
`train_word2vec(corpus, vector_size=..., window=..., min_count=..., epochs=..., seed=...)`.

## 4. Notebook sections

1. Packages
2. Load the processed corpus of Part 1
3. Build the search engine (our 5 test queries and the 2 validation queries)
4. TF-IDF + cosine similarity
5. BM25 (checked with the "president lincoln" example of the slides)
6. Our score (and the numbers of the corpus behind it)
7. How the rankings differ (top 10 of each method, summary table, weights in "men white sneakers", validation queries)
8. Train word2vec (most similar words of some query terms)
9. Top-20 documents of our 5 queries with word2vec + cosine (and comparison with TF-IDF)
