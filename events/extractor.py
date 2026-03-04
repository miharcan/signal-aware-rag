import json


def extract_event(headline, generator):
    prompt = f"""
    Extract structured event information from this headline.

    Headline: "{headline}"

    Return JSON with:
    {{
        "event_type": "...",
        "company": "...",
        "impact_direction": "positive|negative|neutral"
    }}
    """

    response = generator.generate(prompt)

    try:
        return json.loads(response)
    except Exception:
        return None