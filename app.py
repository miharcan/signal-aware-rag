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

    baseline = pipeline.run(
        query,
        signal_aware=False,
        entity_aware=False,
        graph_aware=False
    )

    signal_doc = pipeline.run(
        query,
        signal_aware=True,
        entity_aware=False,
        graph_aware=False
    )

    signal_entity = pipeline.run(
        query,
        signal_aware=True,
        entity_aware=True,
        graph_aware=False
    )

    signal_entity_graph = pipeline.run(
        query,
        signal_aware=True,
        entity_aware=True,
        graph_aware=True
    )

    print("\n=== BASELINE ===")
    for d in baseline["documents"]:
        print(d["company"], d["growth"])

    print("\n=== SIGNAL DOC LEVEL ===")
    for d in signal_doc["documents"]:
        print(d["company"], d["growth"])

    print("\n=== SIGNAL ENTITY LEVEL ===")
    for d in signal_entity["documents"]:
        print(d["company"], d["growth"])

    print("\n=== SIGNAL ENTITY + GRAPH ===")
    for d in signal_entity_graph["documents"]:
        print(d["company"], d["growth"])

    print("\n######## EVENT QUERY ########")
    event_query = "Which companies were impacted by supply chain disruption?"

    event_entity = pipeline.run(
        event_query,
        signal_aware=False,
        entity_aware=True,
        graph_aware=False
    )

    event_entity_graph = pipeline.run(
        event_query,
        signal_aware=False,
        entity_aware=True,
        graph_aware=True
    )

    print("\n=== EVENT ENTITY (no graph) ===")
    for d in event_entity["documents"]:
        print(d["company"], d["growth"])

    print("\n=== EVENT ENTITY + GRAPH ===")
    for d in event_entity_graph["documents"]:
        print(d["company"], d["growth"])

    print("\n\n=== EVALUATION ===")

    baseline_results = evaluate_mode(
        pipeline,
        mode_name="baseline",
        signal_aware=False,
        entity_aware=False,
        graph_aware=False
    )

    signal_doc_results = evaluate_mode(
        pipeline,
        mode_name="signal_doc",
        signal_aware=True,
        entity_aware=False,
        graph_aware=False
    )

    signal_entity_results = evaluate_mode(
        pipeline,
        mode_name="signal_entity",
        signal_aware=True,
        entity_aware=True,
        graph_aware=False
    )

    signal_entity_graph_results = evaluate_mode(
        pipeline,
        mode_name="signal_entity_graph",
        signal_aware=True,
        entity_aware=True,
        graph_aware=True
    )

    for group in [baseline_results, signal_doc_results, signal_entity_results, signal_entity_graph_results]:
        for r in group:
            print(r)


if __name__ == "__main__":
    main()