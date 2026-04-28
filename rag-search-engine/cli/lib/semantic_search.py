from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re

from .search_utils import EMBEDDINGS_PATH, CACHE_DIR, load_movies

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text.strip():
            raise ValueError("Text cannot be empty or whitespace only")

        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents
        doc_list = []
        for document in documents:
            self.document_map[document['id']] = document
            doc_list.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in documents:
            self.document_map[document['id']] = document

        if os.path.exists(EMBEDDINGS_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        embedding = self.generate_embedding(query)
        similarities = []
        for doc_embedding, document in zip(self.embeddings, self.documents):
            sim = cosine_similarity(embedding, doc_embedding)
            similarities.append((sim, document))
        return sorted(similarities, key=lambda x: x[0], reverse=True)[:limit]

def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def verify_embeddings():
    search = SemanticSearch()
    movies = load_movies()
    embeddings = search.load_or_create_embeddings(movies)
    print(f"Number of docs: {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_text(text):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_query_text(query):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def semantic_search(query, limit=5):
    search = SemanticSearch()
    movies = load_movies()
    search.load_or_create_embeddings(movies)
    results = search.search(query, limit=limit)
    for i, (sim, doc) in enumerate(results, 1):
        print(f"{i}. {doc['title']} (score: {sim:.4f})")
        print(f"  {doc['description'][:100]}...")
        print()

def chunk(text, chunk_size=200, overlap=0):
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size")
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}. {chunk}")

def semantic_chunking(text, max_chunk_size=4, overlap=0):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    step = max_chunk_size - overlap
    starts = list(range(0, len(sentences), step))
    if len(starts) >= 2 and starts[-2] + max_chunk_size >= len(sentences):
        starts.pop()
    chunks = [' '.join(sentences[i:i+max_chunk_size]) for i in starts]
    if len(chunks) >= 2 and chunks[-1] in chunks[-2]:
        chunks.pop()
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{chunk}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
