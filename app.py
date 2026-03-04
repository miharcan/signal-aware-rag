from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.generation.openai_generator import OpenAIGenerator
from src.pipeline import RAGPipeline
from src.evaluation.run_evaluation import evaluate_mode
from src.events.extractor import extract_event
from src.events.knowledge_graph import EventGraph
from src.evaluation.llm_judge import judge_answer
import json

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

    def summarize(results, label):
        valid_consistency = [r["consistency"] for r in results if r["consistency"] is not None]
        valid_event = [r["event_precision"] for r in results if r["event_precision"] is not None]

        print(f"\n=== SUMMARY: {label} ===")

        if valid_consistency:
            print("avg_consistency:", round(sum(valid_consistency)/len(valid_consistency), 3))

        if valid_event:
            print("avg_event_precision:", round(sum(valid_event)/len(valid_event), 3))

    summarize(baseline_results, "BASELINE")
    summarize(signal_doc_results, "SIGNAL_DOC")
    summarize(signal_entity_results, "SIGNAL_ENTITY")
    summarize(signal_entity_graph_results, "SIGNAL_ENTITY_GRAPH")

    judge_result = judge_answer(
        generator,
        signal_entity_graph["query"],
        signal_entity_graph["answer"]
    )

    print("\n=== LLM JUDGE ===")
    try:
        judge_json = json.loads(judge_result)
        print(judge_json)
    except json.JSONDecodeError:
        print(judge_result)


if __name__ == "__main__":
    main()