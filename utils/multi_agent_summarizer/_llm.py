import json
import os
from typing import List, Dict
from groq import Groq


def chat(model: str, messages: List[Dict], max_tokens: int = 1800, temperature: float = 0.1) -> str:
    # Use Groq SDK directly to match chat path behavior and avoid base_url mismatches
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    res = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return res.choices[0].message.content


def safe_json_parse(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None
