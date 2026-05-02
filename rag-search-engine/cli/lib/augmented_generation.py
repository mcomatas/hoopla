from .hybrid_search import HybridSearch

from .search_utils import (
    load_movies,
)

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = 'gemma-3-27b-it'

def rag_command(query: str) -> tuple:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    search_results = hybrid_search.rrf_search(query, k=60, limit=5)

    formatted_docs = "\n".join(
        f"- {r['title']}: {r['document']}" for r in search_results
    )

    prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {formatted_docs}

    Answer:"""

    rag_response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    return (search_results, rag_response)

def summarize_command(query: str, limit=5):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    search_results = hybrid_search.rrf_search(query, k=60, limit=limit)

    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Search results:
    {search_results}

    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    summary_response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    return (search_results, summary_response)

def citations_command(query: str, limit: int=5):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    search_results = hybrid_search.rrf_search(query, k=60, limit=limit)

    prompt = f"""Answer the query below and give information based on the provided documents.

    The answer should be tailored to users of Hoopla, a movie streaming service.
    If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

    Query: {query}

    Documents:
    {search_results}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources in the format [1], [2], etc. when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the provided documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""

    citation_response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    return (search_results, citation_response)

def question_command(question: str, limit: int=5):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    search_results = hybrid_search.rrf_search(question, k=60, limit=limit)

    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

    Question: {question}

    Documents:
    {search_results}

    Instructions:
    - Answer questions directly and concisely
    - Be casual and conversational
    - Don't be cringe or hype-y
    - Talk like a normal person would in a chat conversation

    Answer:"""

    question_response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    return (search_results, question_response)
