from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.generation.openai_generator import OpenAIGenerator
from src.pipeline import RAGPipeline
from src.evaluation.run_evaluation import evaluate_mode
from src.events.extractor import extract_event
from src.events.knowledge_graph import EventGraph
from src.evaluation.llm_judge import judge_answer
from src.utils.print_utils import print_docs, print_section, print_subsection
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

    print_section("GRAPH EVENTS")
    for e in graph.events[:5]:
        print(f"{e['company']:<15} → {e['event_type']}")

    pipeline = RAGPipeline(retriever, generator, graph)

    query = "Which companies are showing positive transaction growth?"

    baseline = pipeline.run(
        query,
        signal_aware=False,
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

    print_section("RETRIEVAL COMPARISON")
    print_subsection("Baseline")
    print_docs(baseline["documents"])

    print_subsection("Signal + Entity")
    print_docs(signal_entity["documents"])

    print_subsection("Signal + Entity + Graph")
    print_docs(signal_entity_graph["documents"])
 
    print_section("EVENT QUERY")
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

    print("Query: supply chain disruption")
    print("\nWithout graph")
    for d in event_entity["documents"]:
        print(d["company"])

    print("\nWith graph")
    for d in event_entity_graph["documents"]:
        print(d["company"])

    print_section("EVALUATION DETAILS")

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
            print(
                f"{r['mode']:20} | "
                f"{r['query'][:35]:35} | "
                f"consistency={r['consistency']} | "
                f"event_precision={r['event_precision']}"
            )

    print_section("EVALUATION SUMMARY")
    
    def summarize(results, label):
        valid_consistency = [r["consistency"] for r in results if r["consistency"] is not None]
        valid_event = [r["event_precision"] for r in results if r["event_precision"] is not None]

        print("\n" + label)

        if valid_consistency:
            print("  consistency:      ", round(sum(valid_consistency)/len(valid_consistency),3))

        if valid_event:
            print("  event precision:  ", round(sum(valid_event)/len(valid_event),3))

    summarize(baseline_results, "Baseline")
    summarize(signal_doc_results, "Signal")
    summarize(signal_entity_graph_results, "Signal + Graph")

    judge_result = judge_answer(
        generator,
        signal_entity_graph["query"],
        signal_entity_graph["answer"]
    )

    print_section("LLM JUDGE")

    if isinstance(judge_result, str):
        try:
            judge_result = json.loads(judge_result)
        except json.JSONDecodeError:
            print(judge_result)
            return

    for k, v in judge_result.items():
        print(f"{k:<18} {v}")


if __name__ == "__main__":
    main()