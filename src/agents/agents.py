class AnalystAgent:
    def analyze(self, documents):
        positives = [d for d in documents if d["growth"] > 0]
        negatives = [d for d in documents if d["growth"] < 0]

        avg_growth = 0
        if documents:
            avg_growth = sum(d["growth"] for d in documents) / len(documents)

        return {
            "positive_companies": [d["company"] for d in positives],
            "negative_companies": [d["company"] for d in negatives],
            "average_growth": round(avg_growth, 3),
        }


class ContextAgent:
    def summarize_context(self, documents):
        sectors = list({d["sector"] for d in documents})
        events = list({d.get("event_type") for d in documents if d.get("event_type")})

        return {
            "sectors": sectors,
            "events": events
        }


class WriterAgent:
    def __init__(self, generator):
        self.generator = generator

    def write(self, query, analysis, context):
        prompt = f"""
You are a financial analyst.

Query:
{query}

Structured Analysis:
{analysis}

Context:
{context}

Write a concise investor-style insight.
"""
        return self.generator.generate(prompt)