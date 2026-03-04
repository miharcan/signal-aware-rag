import json
import numpy as np
import faiss

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
        
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def retrieve(self, query, top_k=5):
        query_emb = self.embedder.embed([query]).astype("float32")
        distances, indices = self.index.search(query_emb, top_k)

        seen = set()
        results = []

        for i in indices[0]:
            doc = self.documents[int(i)]
            doc_id = doc["id"]

            if doc_id not in seen:
                seen.add(doc_id)
                results.append(doc)

            if len(results) == top_k:
                break

        return results

    def retrieve_with_filter(self, query, top_k=5, filter_fn=None):
        query_emb = self.embedder.embed([query]).astype("float32")

        # Step 1: filter corpus first
        filtered_docs = [
            d for d in self.documents
            if not filter_fn or filter_fn(d)
        ]

        if not filtered_docs:
            return []

        # Step 2: embed filtered docs
        texts = [d["text"] for d in filtered_docs]
        embeddings = self.embedder.embed(texts).astype("float32")

        # Step 3: build temporary index
        import faiss
        dim = embeddings.shape[1]
        temp_index = faiss.IndexFlatL2(dim)
        temp_index.add(embeddings)

        # Step 4: search inside filtered space
        k = min(top_k, len(filtered_docs))
        distances, indices = temp_index.search(query_emb, k)

        return [filtered_docs[int(i)] for i in indices[0]]

    def retrieve_entity_aware(self, query, top_k=5, filter_fn=None):
        query_emb = self.embedder.embed([query]).astype("float32")

        # Step 1: pre-filter corpus
        filtered_docs = [
            d for d in self.documents
            if not filter_fn or filter_fn(d)
        ]

        if not filtered_docs:
            return []

        # Step 2: embed filtered docs
        texts = [d["text"] for d in filtered_docs]
        embeddings = self.embedder.embed(texts).astype("float32")

        # Step 3: build temporary index
        import faiss
        dim = embeddings.shape[1]
        temp_index = faiss.IndexFlatL2(dim)
        temp_index.add(embeddings)

        # Step 4: search
        distances, indices = temp_index.search(
            query_emb,
            min(top_k * 5, len(filtered_docs))
        )

        # Step 5: group by company
        best_by_company = {}

        for i in indices[0]:
            doc = filtered_docs[int(i)]
            company = doc["company"]

            if company not in best_by_company:
                best_by_company[company] = doc

            if len(best_by_company) >= top_k:
                break

        return list(best_by_company.values())