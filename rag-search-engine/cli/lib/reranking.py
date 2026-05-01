import os
import json
from time import sleep
from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = 'gemma-3-27b-it'

def llm_rerank_individual(
    query: str, documents: list[dict], limit: int = 5
) -> list[dict]:
    scored_docs = []

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Output ONLY the number in your response, no other text or explanation.

        Score:"""

        response = client.models.generate_content(model=model, contents=prompt)
        score_text = (response.text or "").strip()
        score = int(score_text)
        scored_docs.append({**doc, "individual_score": score})
        sleep(3)

    scored_docs.sort(key=lambda x: x["individual_score"], reverse=True)
    return scored_docs[:limit]

def llm_rerank_batch(
    query: str, documents: list[dict], limit: int = 5
) -> list[dict]:
    doc_list_str = "\n".join(
        f"ID {doc['id']} - {doc['title']}: {doc['document']}"
        for doc in documents
    )

    prompt = f"""Rank the movies listed below by relevance to the following search query.

    Query: "{query}"

    Movies:
    {doc_list_str}

    Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

    For example:
    [75, 12, 34, 2, 1]

    Ranking:"""

    results = client.models.generate_content(model=model, contents=prompt)
    ranking_text = (results.text or "").strip()
    ranking = json.loads(ranking_text)
    sorted_docs = [{**doc, "batch_rank": idx + 1} for idx, doc_id in enumerate(ranking) for doc in documents if doc["id"] == doc_id]
    sorted_docs.sort(key=lambda x: x["batch_rank"])
    return sorted_docs[:limit]

def cross_encoder_rerank(
    query: str, documents: list[dict], limit: int = 5
) -> list[dict]:
    pairs = []
    for doc in documents:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)
    sorted_docs = [
        {**doc, "cross_encoder_score": score}
        for score, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    ]
    return sorted_docs[:limit]


def rerank(
    query: str, documents: list[dict], method: str="batch", limit: int = 5
) -> list[dict]:
    if method == "individual":
        return llm_rerank_individual(query, documents, limit)
    elif method == "batch":
        return llm_rerank_batch(query, documents, limit)
    elif method == "cross_encoder":
        return cross_encoder_rerank(query, documents, limit)
    else:
        return documents[:limit]
