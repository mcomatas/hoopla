from PIL import Image
from sentence_transformers import SentenceTransformer

from .search_utils import load_movies
from .semantic_search import cosine_similarity


class MultimodalSearch:
    def __init__(self, documents: list[dict], model_name: str = "clip-ViT-B-32") -> None:
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        embeddings = self.model.encode([image])
        return embeddings[0]

    def search_with_image(self, image_path: str, limit: int = 5) -> list[dict]:
        image_embedding = self.embed_image(image_path)

        scored = []
        for doc, text_embedding in zip(self.documents, self.text_embeddings):
            similarity = cosine_similarity(image_embedding, text_embedding)
            scored.append({
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "similarity": similarity,
            })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:limit]


def verify_image_embedding(image_path: str) -> None:
    searcher = MultimodalSearch(documents=[])
    embedding = searcher.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str) -> list[dict]:
    documents = load_movies()
    searcher = MultimodalSearch(documents)
    return searcher.search_with_image(image_path)
