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

        data.append({
            "id": i,
            "company": company,
            "sector": sector,
            "growth": growth,
            "text": summary
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    generate_synthetic_dataset("data/synthetic/documents.jsonl")