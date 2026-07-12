"""Minimal local stand-in for the course-provided `utils.utils` module.

The original DeepLearning.AI lab container ships its own `ServeLLM` /
`display_info` / `validate_token` implementations outside the notebook
repo. This version reimplements the same interface on top of
`transformers`, so the notebook/scripts run against any local Hugging Face
model path instead of the course's managed GPU environment.
"""
import gc
import os

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer


def display_info(message: str) -> None:
    print(f"[INFO] {message}")


def validate_token() -> bool:
    """Check that a Hugging Face auth token is available and valid."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        display_info("No HF_TOKEN/HUGGINGFACE_TOKEN found in environment.")
        return False
    try:
        HfApi().whoami(token=token)
        return True
    except Exception as exc:
        display_info(f"Token validation failed: {exc}")
        return False


class ServeLLM:
    """Loads a causal LM and serves text completions.

    Usage:
        with ServeLLM(model_path) as llm:
            response = llm.generate_response(prompt, max_tokens=256)
    """

    _instances = []

    def __init__(self, model_path: str, device: str | None = None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def __enter__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        ServeLLM._instances.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._release()
        if self in ServeLLM._instances:
            ServeLLM._instances.remove(self)
        return False

    def generate_response(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _release(self):
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @classmethod
    def cleanup_all(cls):
        for instance in list(cls._instances):
            instance._release()
        cls._instances.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
