import json
import os
import time
import logging
from typing import List, Dict
from groq import Groq

logger = logging.getLogger(__name__)


def chat(model: str, messages: List[Dict], max_tokens: int = 1800, temperature: float = 0.1, response_format: Dict | None = None) -> str:
    # Use Groq SDK directly to match chat path behavior and avoid base_url mismatches
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    t0 = time.perf_counter()
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if response_format is not None:
        kwargs["response_format"] = response_format
    res = client.chat.completions.create(**kwargs)
    dt = (time.perf_counter() - t0) * 1000
    try:
        usage = getattr(res, 'usage', None)
        in_toks = getattr(usage, 'prompt_tokens', None) if usage else None
        out_toks = getattr(usage, 'completion_tokens', None) if usage else None
        logger.info(f"tx.agent.done model={model} ms={dt:.1f} tokens_in={in_toks} tokens_out={out_toks}")
    except Exception:
        logger.info(f"tx.agent.done model={model} ms={dt:.1f}")
    return res.choices[0].message.content


def safe_json_parse(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None
