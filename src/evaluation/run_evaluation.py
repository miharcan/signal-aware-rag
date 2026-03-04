from src.evaluation.metrics import (
    signal_consistency,
    negative_leakage,
    entity_coverage,
)
from src.evaluation.benchmark import BENCHMARK_QUERIES


def evaluate_mode(pipeline, mode_name, signal_aware, entity_aware, graph_aware=False):
    results = []

    for item in BENCHMARK_QUERIES:
        query = item["query"]
        expected = item["expected_direction"]

        output = pipeline.run(
            query,
            signal_aware=signal_aware,
            entity_aware=entity_aware,
            graph_aware=graph_aware
        )

        docs = output["documents"]

        results.append({
            "mode": mode_name,
            "query": query,
            "consistency": signal_consistency(docs, expected),
            "leakage": negative_leakage(docs, expected),
            "entity_coverage": entity_coverage(docs),
        })

    return results