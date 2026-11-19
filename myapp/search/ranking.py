"""
Part 3: ranking methods for the documents returned by the conjunctive (AND) search.

- "tfidf": TF-IDF + cosine similarity (Part 2, myapp/search/algorithms.py)
- "bm25": BM25 with the formula of the theory (log base 2, k1 = 1.2, b = 0.75)
- "ours": our score = BM25 x title factor x stock factor x value factor
- "word2vec": cosine similarity between the average word2vec vectors of the query and the document

Use build_search_engine(corpus) once and then search(query, engine, method).
"""

import zlib

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from myapp.preprocessing.text_processing import build_terms
from myapp.search.algorithms import conjunctive_search, create_index_tfidf, rank_documents

METHODS = ["tfidf", "bm25", "ours", "word2vec"]

K1 = 1.2
B = 0.75


# ---------------------------------------------------------------- BM25

def bm25_term_weight(freq, idf, doc_len, avg_len, k1=K1, b=B):
    """
    BM25 weight of one query term in one document: idf * (k1 + 1) * tf / (k1 * ((1 - b) + b * L_d / L_ave) + tf).

    Argument:
    freq -- frequency of the term in the document
    idf -- log2(N / df) of the term
    doc_len -- number of terms of the document (L_d)
    avg_len -- average number of terms of the documents (L_ave)
    k1, b -- BM25 parameters (tf saturation and length normalization)

    Returns:
    weight
    """
    return idf * (k1 + 1) * freq / (k1 * ((1 - b) + b * doc_len / avg_len) + freq)


def create_bm25_stats(corpus, index, field="search_tokens"):
    """
    Frequencies and lengths needed by BM25 (the idf is the same as in TF-IDF).

    Argument:
    corpus -- processed corpus (pid -> document)
    index -- inverted index of create_index_tfidf
    field -- indexed field

    Returns:
    term_freq -- term -> {pid: frequency}, the frequency is the number of positions in the index
    doc_terms -- pid -> number of terms of the document
    avg_len -- average number of terms per document
    """
    term_freq = {term: {pid: len(positions) for pid, positions in postings} for term, postings in index.items()}
    doc_terms = {pid: len(doc[field]) for pid, doc in corpus.items()}
    avg_len = sum(doc_terms.values()) / len(doc_terms)
    return term_freq, doc_terms, avg_len


def rank_bm25(query_terms, docs, term_freq, idf, doc_terms, avg_len, k1=K1, b=B):
    """
    Rank documents with BM25: sum over the query terms of bm25_term_weight.

    Argument:
    query_terms -- list of query terms
    docs -- documents to rank
    term_freq, doc_terms, avg_len -- output of create_bm25_stats
    idf -- term -> idf
    k1, b -- BM25 parameters

    Returns:
    list of (pid, score) sorted by score, highest first (ties broken by pid)
    """
    terms = [t for t in set(query_terms) if t in idf]
    scores = []
    for pid in docs:
        score = sum(bm25_term_weight(term_freq[t].get(pid, 0), idf[t], doc_terms[pid], avg_len, k1, b)
                    for t in terms)
        scores.append((pid, score))
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores


# ---------------------------------------------------------------- our score

def percentile_in_results(values, higher_is_better=True):
    """
    Position of each value among the results of the query, from 0 (worst) to 1 (best).
    Equal values get the same percentile; missing values get 0.5 (neutral).

    Argument:
    values -- list of numbers (None if missing)
    higher_is_better -- True for the rating, False for the price

    Returns:
    numpy array of percentiles
    """
    series = pd.Series(values, dtype=float)
    result = np.full(len(series), 0.5)
    known = series.notna()
    if known.sum() > 1:
        ranks = series[known].rank(method="average", ascending=higher_is_better)
        result[known.to_numpy()] = ((ranks - 1) / (known.sum() - 1)).to_numpy()
    return result


def rank_our_score(query_terms, docs, corpus, term_freq, idf, doc_terms, avg_len):
    """
    Our score: BM25(q, d) x title(q, d) x stock(d) x value(q, d). Each factor is between 0.5 and 1,
    so none of them can reduce the text score to less than half.
    - title = 0.5 + 0.5 * (query terms in the title) / (query terms)
    - stock = 1 if the product is available, 0.5 if it is out of stock
    - value = 0.5 + 0.25 * rating percentile + 0.25 * cheapness percentile, both computed among the
      documents returned for the query (missing rating or price = 0.5)

    Argument:
    query_terms -- list of query terms
    docs -- documents to rank (the results of the AND search)
    corpus -- processed corpus (title terms and numeric fields)
    term_freq, idf, doc_terms, avg_len -- BM25 statistics

    Returns:
    list of (pid, score) sorted by score, highest first (ties broken by pid)
    """
    docs = sorted(docs)
    if not docs:
        return []
    bm25 = dict(rank_bm25(query_terms, docs, term_freq, idf, doc_terms, avg_len))
    terms = set(query_terms)
    rating = percentile_in_results([corpus[pid]["meta"]["average_rating"] for pid in docs], higher_is_better=True)
    cheap = percentile_in_results([corpus[pid]["meta"]["selling_price"] for pid in docs], higher_is_better=False)

    scores = []
    for i, pid in enumerate(docs):
        title = 0.5 + 0.5 * len(terms & set(corpus[pid]["tokens"]["title"])) / len(terms)
        stock = 0.5 if corpus[pid]["meta"]["out_of_stock"] else 1.0
        value = 0.5 + 0.25 * rating[i] + 0.25 * cheap[i]
        scores.append((pid, bm25[pid] * title * stock * value))
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores


# ---------------------------------------------------------------- word2vec

def stable_hash(text):
    """Hash used by gensim to initialise the word vectors (Python's hash() changes in every run)."""
    return zlib.crc32(text.encode("utf-8"))


def train_word2vec(corpus, field="search_tokens", vector_size=100, window=5, min_count=2, epochs=20, seed=42):
    """
    Train word2vec (CBOW, gensim) on the terms of the documents. With one worker, a fixed seed and
    stable_hash the vectors are the same in every run.

    Argument:
    corpus -- processed corpus
    field -- terms used as sentences (the same terms as the index)
    vector_size, window, seed -- word2vec parameters
    min_count -- terms that appear fewer times are ignored (they get no vector)
    epochs -- passes over the corpus (more than the default 5, the texts are short)

    Returns:
    trained gensim Word2Vec model
    """
    sentences = [doc[field] for doc in corpus.values()]
    return Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=min_count,
                    epochs=epochs, workers=1, seed=seed, hashfxn=stable_hash)


def text_vector(terms, model):
    """
    Vector of a text: average of the word2vec vectors of its terms (terms without vector are skipped).

    Argument:
    terms -- list of terms
    model -- word2vec model

    Returns:
    numpy array (zeros if no term has a vector)
    """
    vectors = [model.wv[t] for t in terms if t in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def create_doc_vectors(corpus, model, field="search_tokens"):
    """
    Average word2vec vector of every document.

    Argument:
    corpus -- processed corpus
    model -- word2vec model
    field -- terms of the document

    Returns:
    pid -> numpy array
    """
    return {pid: text_vector(doc[field], model) for pid, doc in corpus.items()}


def cosine(u, v):
    """Cosine similarity of two vectors (0 if one of them is the zero vector)."""
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.dot(u, v) / norm) if norm > 0 else 0.0


def rank_word2vec(query_terms, docs, model, doc_vectors):
    """
    Rank documents by the cosine similarity between the query vector and the document vector.

    Argument:
    query_terms -- list of query terms
    docs -- documents to rank
    model -- word2vec model
    doc_vectors -- output of create_doc_vectors

    Returns:
    list of (pid, score) sorted by score, highest first (ties broken by pid)
    """
    query_vector = text_vector(query_terms, model)
    scores = [(pid, cosine(query_vector, doc_vectors[pid])) for pid in docs]
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores


# ---------------------------------------------------------------- search

def build_search_engine(corpus, with_word2vec=True):
    """
    Build everything the four ranking methods need.

    Argument:
    corpus -- processed corpus
    with_word2vec -- False to skip training word2vec (only tfidf, bm25 and ours available)

    Returns:
    dict with the index, the TF-IDF and BM25 statistics and the word2vec model and document vectors
    """
    index, weights, idf, doc_length = create_index_tfidf(corpus)
    term_freq, doc_terms, avg_len = create_bm25_stats(corpus, index)
    engine = {"corpus": corpus, "index": index, "weights": weights, "idf": idf, "doc_length": doc_length,
              "term_freq": term_freq, "doc_terms": doc_terms, "avg_len": avg_len}
    if with_word2vec:
        engine["w2v"] = train_word2vec(corpus)
        engine["doc_vectors"] = create_doc_vectors(corpus, engine["w2v"])
    return engine


def search(query, engine, method="tfidf", top_k=None):
    """
    Pre-process the query, keep the documents that contain all its terms (AND) and rank them.

    Argument:
    query -- raw query text
    engine -- output of build_search_engine
    method -- "tfidf", "bm25", "ours" or "word2vec"
    top_k -- number of results to return (None = all)

    Returns:
    list of (pid, score) sorted by score
    """
    query_terms = build_terms(query)
    docs = conjunctive_search(query_terms, engine["index"])
    e = engine
    if method == "tfidf":
        ranking = rank_documents(query_terms, docs, e["weights"], e["idf"], e["doc_length"])
    elif method == "bm25":
        ranking = rank_bm25(query_terms, docs, e["term_freq"], e["idf"], e["doc_terms"], e["avg_len"])
    elif method == "ours":
        ranking = rank_our_score(query_terms, docs, e["corpus"], e["term_freq"], e["idf"], e["doc_terms"], e["avg_len"])
    elif method == "word2vec":
        ranking = rank_word2vec(query_terms, docs, e["w2v"], e["doc_vectors"])
    else:
        raise ValueError("method must be one of {}".format(METHODS))
    return ranking if top_k is None else ranking[:top_k]
