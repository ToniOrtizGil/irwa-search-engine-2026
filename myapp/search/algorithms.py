"""
Inverted index, conjunctive (AND) search and TF-IDF ranking with the cosine similarity
(vector model, as in the theory of the course).

Weights: w(t, d) = (1 + log2 f(t, d)) * log2(N / df(t)) if f(t, d) > 0, else 0.
Score:   cos(q, d) = sum_t w(t, q) * w(t, d) / (|q| * |d|), with |d| the L2 norm of all the
         tf-idf weights of the document.
"""

import math
from collections import Counter, defaultdict

from myapp.preprocessing.text_processing import build_terms


def tf_idf_weight(freq, idf):
    """
    tf-idf weight of a term: (1 + log2 f) * idf if f > 0, else 0.

    Argument:
    freq -- number of times the term appears in the document (or query)
    idf -- idf of the term

    Returns:
    weight
    """
    if freq == 0:
        return 0.0
    return (1 + math.log2(freq)) * idf


def create_index_tfidf(corpus, field="search_tokens"):
    """
    Build the inverted index and compute df, idf, the tf-idf weights and the length of each document.

    Argument:
    corpus -- processed corpus (dict pid -> document) from Part 1
    field -- list of terms of each document that is indexed (the combined field by default)

    Returns:
    index -- term -> list of [pid, list of positions]
    weights -- term -> {pid: tf-idf weight of the term in the document}
    idf -- term -> log2(N / df)
    doc_length -- pid -> L2 norm of the tf-idf weights of the document
    """
    index = defaultdict(list)
    df = defaultdict(int)
    n_docs = len(corpus)

    # 1. positions of each term in each document -> inverted index and df
    for pid, doc in corpus.items():
        current_page_index = {}
        for position, term in enumerate(doc[field]):
            if term in current_page_index:
                current_page_index[term][1].append(position)
            else:
                current_page_index[term] = [pid, [position]]
        for term, posting in current_page_index.items():
            index[term].append(posting)
            df[term] += 1

    # 2. idf of each term
    idf = {term: math.log2(n_docs / df[term]) for term in df}

    # 3. tf-idf weights (the frequency is the number of positions) and document lengths
    weights = defaultdict(dict)
    squares = defaultdict(float)
    for term, postings in index.items():
        for pid, positions in postings:
            w = tf_idf_weight(len(positions), idf[term])
            weights[term][pid] = w
            squares[pid] += w ** 2
    doc_length = {pid: math.sqrt(squares[pid]) for pid in corpus}

    return index, weights, idf, doc_length


def conjunctive_search(query_terms, index):
    """
    Return the documents that contain ALL the query terms (AND query): intersection of the posting lists.

    Argument:
    query_terms -- list of query terms (already pre-processed)
    index -- inverted index

    Returns:
    docs -- set of pids (empty if one of the terms is not in the index)
    """
    docs = None
    for term in set(query_terms):
        term_docs = set(posting[0] for posting in index.get(term, []))
        docs = term_docs if docs is None else docs & term_docs
        if not docs:
            return set()
    return docs if docs is not None else set()


def rank_documents(query_terms, docs, weights, idf, doc_length):
    """
    Rank documents with the cosine similarity between the tf-idf vectors of the query and the document.

    Argument:
    query_terms -- list of query terms
    docs -- documents to rank
    weights -- term -> {pid: tf-idf weight}
    idf -- term -> idf
    doc_length -- pid -> L2 norm of the document vector

    Returns:
    list of (pid, score) sorted by score, highest first (ties broken by pid)
    """
    query_counts = Counter(t for t in query_terms if t in idf)
    query_weights = {t: tf_idf_weight(f, idf[t]) for t, f in query_counts.items()}
    query_length = math.sqrt(sum(w ** 2 for w in query_weights.values()))
    if query_length == 0:
        return []

    scores = []
    for pid in docs:
        dot = sum(w_q * weights[t].get(pid, 0.0) for t, w_q in query_weights.items())
        length = query_length * doc_length[pid]
        scores.append((pid, dot / length if length > 0 else 0.0))
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores


def search_tf_idf(query, index, weights, idf, doc_length):
    """
    Pre-process the query with build_terms(), keep the documents that contain all the query
    terms and rank them with the cosine similarity of tf-idf vectors.

    Argument:
    query -- raw query text
    index, weights, idf, doc_length -- output of create_index_tfidf

    Returns:
    list of (pid, score) sorted by score
    """
    query_terms = build_terms(query)
    docs = conjunctive_search(query_terms, index)
    return rank_documents(query_terms, docs, weights, idf, doc_length)


def search_in_corpus(query):
    """Placeholder of the template, the web app search will use search_tf_idf in Part 4."""
    return ""
