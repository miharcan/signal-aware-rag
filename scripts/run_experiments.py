import sys
from pathlib import Path
import argparse
import numpy as np

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.generation.openai_generator import OpenAIGenerator
from src.pipeline import RAGPipeline
from src.events.extractor import extract_event
from src.events.knowledge_graph import EventGraph
from src.evaluation.run_evaluation import evaluate_mode


def summarize(results):
    consistency_vals = [r["consistency"] for r in results if r["consistency"] is not None]
    event_vals = [r["event_precision"] for r in results if r["event_precision"] is not None]

    c_mean = np.mean(consistency_vals)
    c_std = np.std(consistency_vals)

    e_mean = np.mean(event_vals)
    e_std = np.std(event_vals)

    return (c_mean, c_std), (e_mean, e_std)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="all",
        choices=["baseline", "signal", "signal_entity", "signal_graph", "all"]
    )

    args = parser.parse_args()

    embedder = Embedder()
    retriever = Retriever.from_jsonl(embedder, "data/synthetic/documents.jsonl")
    generator = OpenAIGenerator()

    graph = EventGraph()

    for doc in retriever.documents:
        headline = doc.get("headline")
        if headline:
            event = extract_event(headline, generator)
            graph.add_event(event)

    pipeline = RAGPipeline(retriever, generator, graph)

    experiment_map = {
        "baseline": ("baseline", False, False, False),
        "signal": ("signal", True, False, False),
        "signal_entity": ("signal_entity", True, True, False),
        "signal_graph": ("signal_graph", True, True, True),
    }

    if args.mode == "all":
        experiments = experiment_map.values()
    else:
        experiments = [experiment_map[args.mode]]

    results = []

    for name, signal, entity, graph_flag in experiments:

        eval_results = evaluate_mode(
            pipeline,
            mode_name=name,
            signal_aware=signal,
            entity_aware=entity,
            graph_aware=graph_flag
        )

        (cons_mean, cons_std), (ev_mean, ev_std) = summarize(eval_results)
        results.append((name, cons_mean, cons_std, ev_mean, ev_std))

    print("\n==============================")
    print("EXPERIMENT RESULTS")
    print("==============================\n")

    print(f"{'mode':20} {'consistency':18} {'event_precision'}")
    print("-"*60)

    for name, cm, cs, em, es in results:
        print(f"{name:20} {cm:.3f} ± {cs:.3f}      {em:.3f} ± {es:.3f}")


if __name__ == "__main__":
    main()