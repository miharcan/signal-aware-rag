from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.generation.openai_generator import OpenAIGenerator
from src.pipeline import RAGPipeline

from dotenv import load_dotenv
load_dotenv()


def main():
    embedder = Embedder()
    retriever = Retriever.from_jsonl(
        embedder,
        "data/synthetic/documents.jsonl"
    )
    generator = OpenAIGenerator()

    pipeline = RAGPipeline(retriever, generator)

    query = "Which companies are showing positive transaction growth?"

    baseline = pipeline.run(query, signal_aware=False)
    signal = pipeline.run(query, signal_aware=True)

    print("\n=== BASELINE ===")
    for d in baseline["documents"]:
        print(d["company"], d["growth"])

    print("\n=== SIGNAL AWARE ===")
    for d in signal["documents"]:
        print(d["company"], d["growth"])

    result = pipeline.run(query)

    print(result["answer"])


if __name__ == "__main__":
    main()