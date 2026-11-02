"""
Rank-based evaluation metrics (binary relevance), following the Evaluation lab and the slides.

All the metrics receive `labels`: the relevance labels (1/0) of the documents returned by the search
engine, in ranking order. `total_relevant` is the number of relevant documents of the query (also the
ones that are not returned); if it is not given, it is the number of relevant documents in `labels`.
"""

import numpy as np


def precision_at_k(labels, k=10):
    """
    P@K: fraction of the top K documents that are relevant.

    Argument:
    labels -- relevance labels of the returned documents, in ranking order
    k -- number of documents to consider

    Returns:
    precision@k
    """
    return np.sum(labels[:k]) / k


def get_total_relevant(labels, total_relevant):
    """Number of relevant documents of the query: total_relevant if given, else the relevant in labels."""
    return np.sum(labels) if total_relevant is None else total_relevant


def recall_at_k(labels, k=10, total_relevant=None):
    """
    R@K: fraction of all the relevant documents that appear in the top K.

    Argument:
    labels -- relevance labels of the returned documents, in ranking order
    k -- number of documents to consider
    total_relevant -- number of relevant documents of the query

    Returns:
    recall@k (0 if the query has no relevant documents)
    """
    total_relevant = get_total_relevant(labels, total_relevant)
    if total_relevant == 0:
        return 0.0
    return np.sum(labels[:k]) / total_relevant


def avg_precision_at_k(labels, k=10, total_relevant=None):
    """
    AP@K: average of P@i at the ranks i <= K of the relevant documents. As in the slides, a
    relevant document that is not retrieved in the top K counts with precision 0, so the sum
    is divided by the total number of relevant documents of the query.

    Argument:
    labels -- relevance labels of the returned documents, in ranking order
    k -- number of documents to consider
    total_relevant -- number of relevant documents of the query

    Returns:
    average precision@k
    """
    total_relevant = get_total_relevant(labels, total_relevant)
    if total_relevant == 0:
        return 0.0
    precisions = []
    n_relevant = 0
    for i in range(min(k, len(labels))):
        if labels[i] == 1:
            n_relevant += 1
            precisions.append(n_relevant / (i + 1))
    return np.sum(precisions) / total_relevant


def f1_at_k(labels, k=10, total_relevant=None):
    """
    F1@K: harmonic mean of P@K and R@K.

    Argument:
    labels -- relevance labels of the returned documents, in ranking order
    k -- number of documents to consider
    total_relevant -- number of relevant documents of the query

    Returns:
    f1@k (0 if precision and recall are both 0)
    """
    p = precision_at_k(labels, k)
    r = recall_at_k(labels, k, total_relevant)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def rr_at_k(labels, k=10):
    """
    Reciprocal rank: 1 / position of the first relevant document in the top K (0 if there is none).

    Argument:
    labels -- relevance labels of the returned documents, in ranking order
    k -- number of documents to consider

    Returns:
    reciprocal rank
    """
    top = np.asarray(labels[:k])  # with a plain list, top == 1 would be a single False
    if np.sum(top) == 0:
        return 0.0
    return 1 / (np.argmax(top == 1) + 1)


def dcg_at_k(labels, k=10):
    """
    DCG@K = sum of rel_i / log2(i + 1) for the top K positions (formula of the slides).

    Argument:
    labels -- relevance labels of the returned documents, in ranking order
    k -- number of documents to consider

    Returns:
    dcg@k
    """
    top = np.asarray(labels[:k], dtype=float)
    discounts = np.log2(np.arange(2, len(top) + 2))
    return np.sum(top / discounts)


def ndcg_at_k(labels, k=10, total_relevant=None):
    """
    NDCG@K = DCG@K / IDCG@K, where IDCG@K is the DCG of the ideal ranking: all the relevant
    documents of the query in the first positions.

    Argument:
    labels -- relevance labels of the returned documents, in ranking order
    k -- number of documents to consider
    total_relevant -- number of relevant documents of the query

    Returns:
    ndcg@k (0 if the query has no relevant documents)
    """
    total_relevant = get_total_relevant(labels, total_relevant)
    idcg = dcg_at_k(np.ones(int(min(total_relevant, k))), k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(labels, k) / idcg


def evaluate_query(labels, k=10, total_relevant=None):
    """
    Compute all the per-query metrics.

    Argument:
    labels -- relevance labels of the returned documents, in ranking order
    k -- number of documents to consider
    total_relevant -- number of relevant documents of the query

    Returns:
    dict metric -> value
    """
    return {"P@{}".format(k): precision_at_k(labels, k),
            "R@{}".format(k): recall_at_k(labels, k, total_relevant),
            "AP@{}".format(k): avg_precision_at_k(labels, k, total_relevant),
            "F1@{}".format(k): f1_at_k(labels, k, total_relevant),
            "RR@{}".format(k): rr_at_k(labels, k),
            "NDCG@{}".format(k): ndcg_at_k(labels, k, total_relevant)}


def mean_metrics(per_query, k=10):
    """
    MAP and MRR: mean of AP@K and RR@K over all the queries.

    Argument:
    per_query -- list of dicts returned by evaluate_query
    k -- the K used in evaluate_query

    Returns:
    dict with MAP@k and MRR@k
    """
    return {"MAP@{}".format(k): np.mean([m["AP@{}".format(k)] for m in per_query]),
            "MRR@{}".format(k): np.mean([m["RR@{}".format(k)] for m in per_query])}
