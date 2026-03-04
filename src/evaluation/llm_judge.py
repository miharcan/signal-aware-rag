def judge_answer(generator, query, answer):
    prompt = f"""
You are an expert evaluator.

Evaluate the following answer.

Query:
{query}

Answer:
{answer}

Score from 0 to 1 for:
- faithfulness (grounded in provided context)
- event_alignment (correctly answers event if present)
- business_quality (useful financial insight)

Return ONLY JSON like:
{{
  "faithfulness": 0.0,
  "event_alignment": 0.0,
  "business_quality": 0.0
}}
"""

    response = generator.generate(prompt)

    return response