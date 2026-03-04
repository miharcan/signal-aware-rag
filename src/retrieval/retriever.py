import json
import numpy as np

class Retriever:
    def __init__(self, embedder, index, documents):
        self.embedder = embedder
        self.index = index
        self.documents = documents

    @classmethod
    def from_jsonl(cls, embedder, jsonl_path):
        with open(jsonl_path) as f:
            docs = [json.loads(line) for line in f]

        texts = [d["text"] for d in docs]
        embeddings = embedder.embed(texts)

        index = cls._build_index(embeddings)
        return cls(embedder, index, docs)

    @staticmethod
    def _build_index(embeddings):
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def retrieve(self, query, top_k=5):
        query_emb = self.embedder.embed([query]).astype("float32")
        indices, _ = self.index.search(query_emb, top_k)
        return [self.documents[int(i)] for i in indices[0]]
