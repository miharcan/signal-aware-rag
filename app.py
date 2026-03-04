from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.generation.openai_generator import OpenAIGenerator
from src.pipeline import RAGPipeline

def main():
    embedder = Embedder()
    retriever = Retriever.from_jsonl(
        embedder,
        "data/synthetic/documents.jsonl"
    )
    generator = OpenAIGenerator()

    pipeline = RAGPipeline(retriever, generator)

    query = "Which companies are showing positive transaction growth?"
    result = pipeline.run(query)

    print(result["answer"])

if __name__ == "__main__":
    main()