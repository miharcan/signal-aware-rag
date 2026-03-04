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

    for i in range(n_docs):
        company = random.choice(COMPANIES)
        sector = random.choice(SECTORS)
        growth = round(random.uniform(-0.1, 0.15), 3)

        summary = (
            f"{company} reported weekly transaction growth of "
            f"{growth*100:.1f}% in the {sector} sector. "
            f"Consumer demand trends remain volatile."
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