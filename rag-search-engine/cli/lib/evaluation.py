from .hybrid_search import HybridSearch
from .search_utils import (
    load_golden_dataset,
    load_movies
)

def precision_at_k(
    retrieved_docs: list[str],
    relevant_docs: set[str],
    k: int = 5
) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / k

def recall_at_k(
    retrieved_docs: list[str],
    relevant_docs: set[str],
    k: int = 5
) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / len(relevant_docs)

def f1_score(
    precision: float,
    recall: float
) -> float:
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_golden_dataset(limit: int = 5) -> dict:
    golden_dataset = load_golden_dataset()
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = {}
    total_precision = 0
    for test_case in golden_dataset:
        relevant_docs = set(test_case["relevant_docs"])
        query = test_case["query"]
        search_results = hybrid_search.rrf_search(query, k=60, limit=limit)
        retrieved_docs = []
        for result in search_results:
            title = result.get("title", "")
            if title:
                retrieved_docs.append(title)

        precision = precision_at_k(retrieved_docs, relevant_docs, limit)
        total_precision += precision
        recall = recall_at_k(retrieved_docs, relevant_docs, limit)
        f1 = f1_score(precision, recall)

        results[query] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "retrieved": retrieved_docs[:limit],
            "relevant": list(relevant_docs)
        }

    return results
