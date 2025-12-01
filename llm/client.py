# llm/client.py
import os
from typing import List, Dict, Any
import httpx

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "dummy")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instruct")


class LLMClient:
    """
    Thin wrapper around an OpenAI-compatible local model endpoint.
    Keep this minimal; behavior is driven by prompts, not code.
    """

    def __init__(self):
        self._client = httpx.Client(
            base_url=LLM_BASE_URL,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"}
        )

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        payload = {
            "model": LLM_MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
        }
        resp = self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
