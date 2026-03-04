import re


def extract_event_from_query(query: str):
    query = query.lower()

    # Remove punctuation
    query = re.sub(r"[^\w\s]", "", query)

    query = query.replace(" ", "_")

    KNOWN_EVENTS = [
        "supply_chain_disruption",
        "demand_shock",
        "pricing_change",
        "guidance_revision",
        "regulatory_event",
    ]

    for event in KNOWN_EVENTS:
        if event in query:
            return event

    return None