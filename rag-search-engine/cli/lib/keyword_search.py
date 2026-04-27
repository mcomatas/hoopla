import string
import pickle
import os
import sys
import math

from nltk.stem import PorterStemmer
from collections import Counter

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    CACHE_DIR,
    INDEX_PATH,
    DOCMAP_PATH,
    FREQUENCIES_PATH,
    DOC_LENGTHS_PATH,
    BM25_K1,
    BM25_B,
    load_movies,
    load_stopwords
)

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}

    def __add_document(self, doc_id, text):
        tokens = tokenization(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter()
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0

        total_length = 0
        for length in self.doc_lengths.values():
            total_length += length
        return total_length / len(self.doc_lengths)

    def get_documents(self, term):
        term = term.lower()
        return sorted(self.index.get(term, set()))

    def get_tf(self, doc_id, term):
        term = tokenization(term)
        if len(term) > 1:
            raise Exception("Term must be a single token")
        return self.term_frequencies.get(doc_id, Counter()).get(term[0], 0)

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenization(term)
        if len(tokens) > 1:
            raise Exception("Term must be a single token")
        df = len(self.index.get(tokens[0], set()))
        return math.log((len(self.docmap) - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        tf = self.get_tf(doc_id, term)
        saturated_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return saturated_tf

    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit):
        query_tokens = tokenization(query)
        scores: dict[int, float] = {}
        for doc_id in self.docmap:
            score = 0.0
            for term in query_tokens:
                score += self.bm25(doc_id, term)
            scores[doc_id] = score
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

    def build(self):
        movies = load_movies()
        for movie in movies:
            # id, title, description
            self.__add_document(movie['id'], f"{movie['title']} {movie['description']}")
            self.docmap[movie['id']] = movie

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(INDEX_PATH, 'wb') as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_PATH, 'wb') as f:
            pickle.dump(self.docmap, f)
        with open(FREQUENCIES_PATH, 'wb') as f:
            pickle.dump(self.term_frequencies, f)
        with open(DOC_LENGTHS_PATH, 'wb') as f:
            pickle.dump(self.doc_lengths, f)


    def load(self):
        with open(INDEX_PATH, 'rb') as f:
            self.index = pickle.load(f)
        with open(DOCMAP_PATH, 'rb') as f:
            self.docmap = pickle.load(f)
        with open(FREQUENCIES_PATH, 'rb') as f:
            self.term_frequencies = pickle.load(f)
        with open(DOC_LENGTHS_PATH, 'rb') as f:
            self.doc_lengths = pickle.load(f)


def bm25search_command(query: str, limit: int=5) -> list[tuple[int, float]]:
    idx = InvertedIndex()
    idx.load()
    results = idx.bm25_search(query, limit)
    return [(doc_id, idx.docmap[doc_id]["title"], score) for doc_id, score in results]

def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def tf_idf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    tokens = tokenization(term)
    if len(tokens) != 1:
        raise Exception("Term must be a single token")
    tf = idx.get_tf(doc_id, tokens[0])
    idf = idf_command(term)
    return tf * idf


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    tokens = tokenization(term)
    if len(tokens) != 1:
        raise Exception("Term must be a single token")
    df = len(idx.index.get(tokens[0], set()))
    return math.log((len(idx.docmap) + 1) / (df + 1))

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
        query_tokens = tokenization(query)
        seen, results = set(), []
        for q_token in query_tokens:
            for doc_id in inverted_index.get_documents(q_token):
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                results.append(inverted_index.docmap[doc_id])
                if len(results) >= limit:
                    return results
        return results
    except FileNotFoundError:
        print("Index not found. Run 'build' first.", file=sys.stderr)
        sys.exit(1)

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenization(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stopwords = load_stopwords()
    words = remove_stopwords(valid_tokens, stopwords)
    return stem_words(words)

def tokenization_match(q_tokens: list[str], t_tokens: list[str]) -> bool:
    for q_token in q_tokens:
        for t_token in t_tokens:
            if q_token in t_token:
                return True
    return False

def remove_stopwords(tokens: list[str], stopwords: list[str]) -> list[str]:
    return [token for token in tokens if token not in stopwords]

def stem_words(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]
