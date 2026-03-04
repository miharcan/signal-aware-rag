from src.events.query_event import extract_event_from_query


def evaluate_mode(
    pipeline,
    mode_name,
    signal_aware,
    entity_aware,
    graph_aware=False
):
    benchmark = [
        {
            "query": "Which companies are showing positive transaction growth?",
            "expected_direction": "positive",
        },
        {
            "query": "Which companies are experiencing negative transaction growth?",
            "expected_direction": "negative",
        },
        {
            "query": "List companies with positive growth in Technology.",
            "expected_direction": "positive",
        },
        {
            "query": "Which companies were impacted by supply chain disruption?",
            "expected_direction": None,
        },
    ]

    results = []

    for item in benchmark:
        query = item["query"]
        expected_direction = item["expected_direction"]

        output = pipeline.run(
            query,
            signal_aware=signal_aware,
            entity_aware=entity_aware,
            graph_aware=graph_aware
        )

        docs = output["documents"]
        returned_companies = {d["company"] for d in docs}

        # -------------------------
        # Growth Consistency Metric
        # -------------------------
        consistency = None
        leakage = None

        if expected_direction:
            if returned_companies:
                correct = 0
                wrong = 0

                for d in docs:
                    if expected_direction == "positive" and d["growth"] > 0:
                        correct += 1
                    elif expected_direction == "negative" and d["growth"] < 0:
                        correct += 1
                    else:
                        wrong += 1

                consistency = correct / len(docs)
                leakage = wrong / len(docs)
            else:
                consistency = 0.0
                leakage = 0.0

        # -------------------------
        # Event Precision Metric
        # -------------------------
        query_event = extract_event_from_query(query)
        event_precision = None

        if query_event:
            affected = pipeline.graph.query_by_event_type(query_event)
            true_companies = {e["company"] for e in affected}

            if returned_companies:
                event_precision = (
                    len(returned_companies & true_companies)
                    /
                    len(returned_companies)
                )
            else:
                event_precision = 0.0

        # -------------------------
        # Entity Coverage
        # -------------------------
        entity_coverage = len(returned_companies)

        results.append({
            "mode": mode_name,
            "query": query,
            "consistency": consistency,
            "leakage": leakage,
            "entity_coverage": entity_coverage,
            "event_precision": event_precision,
        })

    return results