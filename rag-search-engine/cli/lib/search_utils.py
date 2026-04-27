import json
import os

DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MOVIES_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT,"cache")
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
FREQUENCIES_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")
DOC_LENGTHS_PATH = os.path.join(CACHE_DIR, "doc_lengths.pkl")


def load_movies() -> list[dict]:
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        data = f.read()
    return data.splitlines()
