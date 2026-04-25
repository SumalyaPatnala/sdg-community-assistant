import os
import requests
from serving.prompts import build_prompt
from serving.finetuned_model import FineTunedModel


class ModelRouter:
    """
    Routes inference between local and remote models.

    Local:
    - Ollama on localhost
    - Example: ollama run phi3

    Remote:
    - Hugging Face Inference API or any hosted endpoint
    """

    def __init__(self):
        self.finetuned_model = FineTunedModel()
        self.local_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.local_model = os.getenv("LOCAL_MODEL", "phi3")

        self.remote_url = os.getenv(
            "HF_API_URL",
            "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
        )
        self.hf_token = os.getenv("HF_TOKEN", "")
        self.timeout = int(os.getenv("INFERENCE_TIMEOUT", "120"))

    def local_available(self) -> bool:
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            return response.status_code == 200
        except Exception:
            return False

    def remote_available(self) -> bool:
        return bool(self.remote_url)

    def model_registry(self):
        
        return {
            "local": {
                "name": self.local_model,
                "runtime": "Ollama",
                "url": self.local_url
            },
            "remote": {
                "name": "Phi-3-mini-4k-instruct",
                "runtime": "Hugging Face Inference API",
                "url": self.remote_url
            },
            "fine_tuned": {
            "base_model": self.finetuned_model.base_model,
            "adapter_path": self.finetuned_model.adapter_path,
            "available": self.finetuned_model.available()
            },
            "routing_policy": "auto -> fine-tuned first, then local, then remote fallback"
        }

    def generate(self, task: str, text: str, model_preference: str = "auto", max_tokens: int = 512):
        prompt = build_prompt(task=task, text=text)

        if model_preference == "finetuned":
            if self.finetuned_model.available():
                return self.finetuned_model.generate(prompt, max_tokens), "finetuned:lora-adapter"
            raise RuntimeError("Fine-tuned adapter not found.")

        if model_preference == "local":
            return self._generate_local(prompt, max_tokens), f"local:{self.local_model}"

        if model_preference == "remote":
            return self._generate_remote(prompt, max_tokens), "remote:huggingface"

        # Auto mode: fine-tuned first, then local, then remote.
        try:
            if self.finetuned_model.available():
                return self.finetuned_model.generate(prompt, max_tokens), "finetuned:lora-adapter"
        except Exception:
            pass

        try:
            if self.local_available():
                return self._generate_local(prompt, max_tokens), f"local:{self.local_model}"
        except Exception:
            pass

        return self._generate_remote(prompt, max_tokens), "remote:huggingface"

    def _generate_local(self, prompt: str, max_tokens: int):
        payload = {
            "model": self.local_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.2
            }
        }
        response = requests.post(self.local_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json().get("response", "")

    def _generate_remote(self, prompt: str, max_tokens: int):
        headers = {}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.2,
                "return_full_text": False
            }
        }

        response = requests.post(self.remote_url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list) and data:
            return data[0].get("generated_text", str(data[0]))
        if isinstance(data, dict):
            return data.get("generated_text", str(data))
        return str(data)
