def signal_consistency(docs, expected_direction):
    if not docs:
        return 0.0

    correct = 0

    for d in docs:
        if expected_direction == "positive" and d["growth"] > 0:
            correct += 1
        elif expected_direction == "negative" and d["growth"] < 0:
            correct += 1

    return correct / len(docs)


def negative_leakage(docs, expected_direction):
    if not docs:
        return 0.0

    violations = 0

    for d in docs:
        if expected_direction == "positive" and d["growth"] <= 0:
            violations += 1
        elif expected_direction == "negative" and d["growth"] >= 0:
            violations += 1

    return violations / len(docs)


def entity_coverage(docs):
    return len(set(d["company"] for d in docs))