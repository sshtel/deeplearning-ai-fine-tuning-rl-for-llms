"""Step 1: Setup - imports and model paths.

Uses the local `utils/utils.py` module (ServeLLM, display_info, validate_token)
at the repo root. Model/data paths default to the course container's
/app/models and /app/data, but can be overridden via environment variables
for local testing against any Hugging Face model path or cached dataset.
"""
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.utils import ServeLLM, display_info, validate_token  # noqa: E402
from datasets import load_from_disk  # noqa: E402

warnings.filterwarnings("ignore")

# Model definitions (override via env vars for local testing)
BASE_MODEL = os.environ.get("BASE_MODEL", "/app/models/deepseek-math-7b-base")
SFT_MODEL = os.environ.get("SFT_MODEL", "/app/models/deepseek-math-7b-instruct")
RL_MODEL = os.environ.get("RL_MODEL", "/app/models/deepseek-math-7b-rl")
LLAMA_GUARD = os.environ.get("LLAMA_GUARD_MODEL", "/app/models/Llama-Guard-3-8B")

if __name__ == "__main__":
    print("All warnings suppressed.")
    print("Setup complete! Ready to start the lab.")
