class QuerySignals:
    def __init__(self, growth_direction=None, sector=None):
        self.growth_direction = growth_direction
        self.sector = sector


def extract_signals(query: str) -> QuerySignals:
    query_lower = query.lower()

    growth_direction = None
    sector = None

    if "positive" in query_lower or "increase" in query_lower:
        growth_direction = "positive"
    elif "negative" in query_lower or "decline" in query_lower:
        growth_direction = "negative"

    if "technology" in query_lower:
        sector = "Technology"
    elif "healthcare" in query_lower:
        sector = "Healthcare"
    elif "consumer" in query_lower:
        sector = "Consumer Discretionary"

    return QuerySignals(
        growth_direction=growth_direction,
        sector=sector
    )