class RAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def run(self, query: str, top_k: int = 5):
        docs = self.retriever.retrieve(query, top_k)

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
            "documents": docs
        }