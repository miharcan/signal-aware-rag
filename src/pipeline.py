from src.signals.query_signals import extract_signals


class RAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def run(self, query: str, top_k: int = 5, signal_aware: bool = False):
        if not signal_aware:
            docs = self.retriever.retrieve(query, top_k)
        else:
            signals = extract_signals(query)

            def filter_fn(doc):
                if signals.growth_direction == "positive" and doc["growth"] <= 0:
                    return False
                if signals.growth_direction == "negative" and doc["growth"] >= 0:
                    return False
                if signals.sector and doc["sector"] != signals.sector:
                    return False
                return True

            docs = self.retriever.retrieve_with_filter(
                query,
                top_k=top_k,
                filter_fn=filter_fn
            )

        context = "\n".join([d["text"] for d in docs])

        prompt = f"""
        Use the context below to answer the question.

        Context:
        {context}

        Question:
        {query}
        """

        answer = self.generator.generate(prompt)

        return {
            "query": query,
            "answer": answer,
            "documents": docs,
            "signal_aware": signal_aware
        }