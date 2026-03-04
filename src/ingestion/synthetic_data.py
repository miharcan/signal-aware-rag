import random
import json
import os

EVENT_MAP = {
    "Alpha Retail": "demand_shock",
    "Beta Apparel": "regulatory_event",
    "Gamma Tech": "pricing_change",
    "Delta Health": "supply_chain_disruption",
    "Epsilon Stores": "guidance_revision",
}

SECTORS = [
    "Consumer Discretionary",
    "Healthcare",
    "Technology"
]

companies = list(EVENT_MAP.keys())

documents = []

for i, company in enumerate(companies):

    sector = random.choice(SECTORS)
    growth = round(random.uniform(-0.15, 0.15), 3)

    event_type = EVENT_MAP[company]
    event_readable = event_type.replace("_", " ")

    headline = f"{event_readable} affects {company}"
    summary = f"Weekly spend trends for {company} shifted by {growth*100:.1f}% within the {sector} sector."

    text = f"{headline}. {summary}"

    documents.append({
        "id": i,
        "company": company,
        "sector": sector,
        "growth": growth,
        "headline": headline,
        "event_type": event_type,
        "text": text
    })

output_path = "data/synthetic/documents.jsonl"

# Ensure directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    for doc in documents:
        f.write(json.dumps(doc) + "\n")

print(f"Saved {len(documents)} documents to {output_path}")