import json
import re

def extract_event(headline, generator):
    prompt = f"""
    Extract structured event information from this headline.

    Headline: "{headline}"

    Return ONLY valid JSON with:
    {{
        "event_type": "...",
        "company": "...",
        "impact_direction": "positive|negative|neutral"
    }}
    """

    response = generator.generate(prompt)

    # Remove markdown fences
    response = response.strip()
    response = re.sub(r"^```json", "", response)
    response = re.sub(r"```$", "", response)
    response = response.strip()

    try:
        event = json.loads(response)
        event["event_type"] = normalize_event_type(event["event_type"])
        return event
    except Exception:
        print("FAILED TO PARSE:", response)
        return None


def normalize_event_type(event_type):
    event_type = event_type.lower().replace(" ", "_")

    mapping = {
        "demand_shock": "demand_shock",
        "demand shock": "demand_shock",
        "supply_chain_disruption": "supply_chain_disruption",
        "supply chain disruption": "supply_chain_disruption",
        "regulatory action": "regulatory_event",
        "regulatory_event": "regulatory_event",
        "regulatoryevent": "regulatory_event",
        "pricing change": "pricing_change",
        "pricing_change": "pricing_change",
        "guidance revision": "guidance_revision",
        "guidance_revision": "guidance_revision",
        "revision of forward guidance": "guidance_revision",
        "revised forward guidance": "guidance_revision",
        "regulatory_action": "regulatory_event",
    }

    return mapping.get(event_type, event_type)