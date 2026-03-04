from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.generation.openai_generator import OpenAIGenerator
from src.pipeline import RAGPipeline
from src.evaluation.run_evaluation import evaluate_mode
from src.events.extractor import extract_event
from src.events.knowledge_graph import EventGraph


from dotenv import load_dotenv
load_dotenv()


def main():
    embedder = Embedder()
    retriever = Retriever.from_jsonl(
        embedder,
        "data/synthetic/documents.jsonl"
    )
    generator = OpenAIGenerator()

    graph = EventGraph()

    for doc in retriever.documents:
        headline = doc.get("headline")
        if not headline:
            continue

        event = extract_event(headline, generator)
        graph.add_event(event)

    print("\n=== SAMPLE GRAPH EVENTS ===")
    for e in graph.events[:5]:
        print(e)

    pipeline = RAGPipeline(retriever, generator, graph)

    query = "Which companies are showing positive transaction growth?"

    baseline = pipeline.run(query, signal_aware=False)

    signal_doc = pipeline.run(
        query,
        signal_aware=True,
        entity_aware=True
    )

    signal_entity = pipeline.run(
        query,
        signal_aware=True,
        entity_aware=True
    )

    print(signal_entity["documents"])
    print(signal_entity["answer"]) 

    print("\n=== BASELINE ===")
    for d in baseline["documents"]:
        print(d["company"], d["growth"])

    print("\n=== SIGNAL DOC LEVEL ===")
    for d in signal_doc["documents"]:
        print(d["company"], d["growth"])

    print("\n=== SIGNAL ENTITY LEVEL ===")
    for d in signal_entity["documents"]:
        print(d["company"], d["growth"])

    result = pipeline.run(query)

    print(result["answer"])

    print("\n\n=== EVALUATION ===")

    baseline_results = evaluate_mode(
        pipeline,
        mode_name="baseline",
        signal_aware=False,
        entity_aware=False
    )

    signal_doc_results = evaluate_mode(
        pipeline,
        mode_name="signal_doc",
        signal_aware=True,
        entity_aware=False
    )

    signal_entity_results = evaluate_mode(
        pipeline,
        mode_name="signal_entity",
        signal_aware=True,
        entity_aware=True
    )

    for group in [baseline_results, signal_doc_results, signal_entity_results]:
        for r in group:
            print(r)


if __name__ == "__main__":
    main()