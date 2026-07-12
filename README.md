# deeplearning-ai-fine-tuning-rl-for-llms

## M1_G1: Inspecting Fine-Tuned vs Base Model

Graded lab comparing three stages of the DeepSeek Math model training pipeline — **Base**, **Fine-Tuned (SFT)**, and **RL** — plus a safety evaluation using Llama Guard.

### Steps

1. **Setup** — Import dependencies (`ServeLLM` wrapper, `pandas`, `datasets`, etc.) and define paths to the four models: base, SFT, RL, and Llama Guard.
2. **Example Prompts** — Define a small set of math test prompts with expected keyword answers for quick qualitative comparison.
3. **Processing Function (Exercise 1)** — Implement `process_prompts()` to generate responses from a given model via `llm.generate_response()`. Run it against the base, SFT, and RL models to compare response style and quality.
4. **Response Scoring and Evaluation** — `score_response()` / `score_all_responses()` check whether expected keywords appear in each model's response, producing a comparison table and average scores across the three models.
5. **GSM8K Dataset** — Load the GSM8K grade-school math word problem dataset for a larger, more rigorous evaluation.
6. **Extract Number (Exercise 2)** — Implement `extract_number()` to parse a numeric answer from model output, first matching the `#### number` GSM8K format, then falling back to the last number in the text.
7. **Model Evaluation (Exercise 3)** — Implement `evaluate_model_correctness()` to generate answers for GSM8K problems, extract and compare model vs. gold answers, and compute accuracy. Run on a small sample, then across all three models (Base, SFT, RL) to compare mathematical reasoning accuracy.
8. **Safety Evaluation** — Load a harmful-prompt dataset (JailbreakBench) plus a set of benign prompts, and introduce Llama Guard's safety categories (S1–S13).
9. **Parse Llama Guard Response (Exercise 4)** — Implement `parse_llama_guard_response()` to turn raw Llama Guard text output into a structured `{classification, categories}` dict.
10. **Safety Metrics (Exercise 5)** — Implement `calculate_safety_metrics()` to compute harmful detection rate, benign acceptance rate, false positive rate, and false negative rate.
11. **Category Analysis & Full Pipeline** — `analyze_safety_categories()` ranks violation categories by frequency; `evaluate_safety_model()` runs the full pipeline (generate → parse → score) over harmful and benign prompts using Llama Guard, then reports the resulting metrics.
12. **Cleanup** — Release GPU memory with `ServeLLM.cleanup_all()`.

### Key Takeaway

The lab illustrates how each post-training stage (base → SFT → RL) changes response structure and task accuracy, and separately how a dedicated safety model (Llama Guard) can be evaluated for harmful-content detection vs. false-positive tradeoffs.

### Code

Each step above is implemented as a standalone, runnable module under [src/](src/):

| File | Step |
|---|---|
| [src/step01_setup.py](src/step01_setup.py) | 1. Setup |
| [src/step02_example_prompts.py](src/step02_example_prompts.py) | 2. Example Prompts |
| [src/step03_process_prompts.py](src/step03_process_prompts.py) | 3. Processing Function (Exercise 1) |
| [src/step04_scoring.py](src/step04_scoring.py) | 4. Response Scoring and Evaluation |
| [src/step05_gsm8k_dataset.py](src/step05_gsm8k_dataset.py) | 5. GSM8K Dataset |
| [src/step06_extract_number.py](src/step06_extract_number.py) | 6. Extract Number (Exercise 2) |
| [src/step07_model_evaluation.py](src/step07_model_evaluation.py) | 7. Model Evaluation (Exercise 3) |
| [src/step08_safety_setup.py](src/step08_safety_setup.py) | 8. Safety Evaluation setup |
| [src/step09_parse_llama_guard.py](src/step09_parse_llama_guard.py) | 9. Parse Llama Guard Response (Exercise 4) |
| [src/step10_safety_metrics.py](src/step10_safety_metrics.py) | 10. Safety Metrics (Exercise 5) |
| [src/step11_safety_pipeline.py](src/step11_safety_pipeline.py) | 11. Category Analysis & Full Pipeline |
| [src/step12_cleanup.py](src/step12_cleanup.py) | 12. Cleanup |

[main.py](main.py) wires all steps together into a single end-to-end run: `python main.py`.

Each file also runs standalone (e.g. `python src/step06_extract_number.py`) and, where relevant, doubles as its own smoke test.

`step06_extract_number.py`, `step09_parse_llama_guard.py`, and `step10_safety_metrics.py` have no external dependencies beyond the standard library and can be run standalone anywhere. Every other step needs [utils/utils.py](utils/utils.py) and real model paths — see below.

### Setup

1. Create and activate a venv (pyenv 3.12+), then install dependencies from [pyproject.toml](pyproject.toml):
   ```
   pip install -e .
   ```
2. `step01_setup.py` and everything downstream of it import [utils/utils.py](utils/utils.py) — a local reimplementation of the course-only `ServeLLM` / `display_info` / `validate_token` module, built on `transformers`/`torch` (with `cuda` device support; `mps`/Apple Silicon not yet wired in).
3. Model paths default to the course container's `/app/models/...`. Override them via env vars to point at real Hugging Face repo ids or local paths:
   ```
   BASE_MODEL=... SFT_MODEL=... RL_MODEL=... LLAMA_GUARD_MODEL=... python3 main.py
   ```
   The DeepSeek-Math-7B models are public; `Llama-Guard-3-8B` is gated on Hugging Face and needs an accepted license + `HF_TOKEN` set.
4. GSM8K and JailbreakBench datasets are loaded via `load_from_disk("/app/data/...")` (`step05_gsm8k_dataset.py`, `step08_safety_setup.py`), matching the course container layout — point these at local dataset directories to run outside that environment.

### Verified

Ran `main.py` directly: it imports and executes cleanly, failing only where expected — at model load time, since `/app/models/...` isn't mounted here. Separately smoke-tested the full pipeline (`ServeLLM` load → `process_prompts` → scoring → `evaluate_model_correctness` → `parse_llama_guard_response` → `evaluate_safety_model`/`calculate_safety_metrics` → `analyze_safety_categories` → `cleanup`) end-to-end against a tiny public model (`sshleifer/tiny-gpt2`) and small in-memory datasets — every stage ran without errors. Accuracy/safety numbers from that smoke test are meaningless (untrained tiny model); it only confirms the code paths are correct.
