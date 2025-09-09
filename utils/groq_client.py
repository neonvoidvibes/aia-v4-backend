import os
from openai import OpenAI


def groq_openai_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("GROQ_OPENAI_BASE_URL", "https://api.groq.com/openai"),
        api_key=os.getenv("GROQ_API_KEY"),
    )


def std_model() -> str:
    return os.getenv("GROQ_MODEL_STD", "openai/gpt-oss-20b")


def integ_model() -> str:
    return os.getenv("GROQ_MODEL_INTEGRATION", "openai/gpt-oss-120b")

