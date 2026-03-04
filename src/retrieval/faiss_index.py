import faiss
import numpy as np

class FaissIndex:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)

    def add(self, embeddings: np.ndarray):
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        distances, indices = self.index.search(query_embedding, top_k)
        return indices[0], distances[0]