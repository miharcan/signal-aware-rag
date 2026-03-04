import json
import random
from pathlib import Path

SECTORS = ["Consumer Discretionary", "Technology", "Healthcare"]

COMPANIES = [
    "Alpha Retail",
    "Beta Apparel",
    "Gamma Tech",
    "Delta Health",
    "Epsilon Stores",
]

EVENT_TYPES = [
    "supply_chain_disruption",
    "demand_shock",
    "pricing_change",
    "guidance_revision",
    "regulatory_event",
]

def generate_synthetic_dataset(output_path: str, n_docs: int = 50):
    data = []
    random.seed(42)

    for i in range(n_docs):
        company = random.choice(COMPANIES)
        sector = random.choice(SECTORS)
        if i % 2 == 0:
            growth = round(random.uniform(0.01, 0.15), 3)
        else:
            growth = round(random.uniform(-0.15, -0.01), 3)

        templates = [
            "{company} saw transaction growth of {growth}% this week in {sector}.",
            "Weekly spend trends for {company} shifted by {growth}% within the {sector} sector.",
            "{company} reported a {growth}% change in consumer transactions across {sector}.",
            "In {sector}, {company} experienced {growth}% week-over-week transaction movement."
        ]

        template = random.choice(templates)

        summary = template.format(
            company=company,
            growth=round(growth * 100, 1),
            sector=sector
        )

        event_type = random.choice(EVENT_TYPES)

        headline_templates = {
            "supply_chain_disruption": "Supply chain disruption impacts {company}",
            "demand_shock": "Unexpected demand shock affects {company}",
            "pricing_change": "{company} announces major pricing change",
            "guidance_revision": "{company} revises forward guidance",
            "regulatory_event": "New regulatory action targets {company}",
        }

        headline = headline_templates[event_type].format(company=company)

        data.append({
            "id": i,
            "company": company,
            "sector": sector,
            "growth": growth,
            "text": summary,
            "headline": headline,
            "event_type": event_type
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    generate_synthetic_dataset("data/synthetic/documents.jsonl")