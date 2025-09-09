import json
from typing import List, Dict
from utils.groq_client import groq_openai_client


def chat(model: str, messages: List[Dict], max_tokens: int = 1800, temperature: float = 0.1) -> str:
    client = groq_openai_client()
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

