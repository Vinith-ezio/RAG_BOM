"""
Qwen Client
-----------
Simple Ollama client for local Qwen inference.
"""

from typing import Optional

import requests


class QwenClient:
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> bool:
        """Check whether Ollama is running."""

        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )

            return response.status_code == 200

        except requests.RequestException:
            return False

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate a response from Qwen through Ollama.
        """

        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        return data.get("response", "").strip()