"""Run the full M1_G1 pipeline end-to-end: base vs SFT vs RL vs safety evaluation.

Mirrors M1_G1_Inspecting_Finetuned_vs_Base_Model.ipynb, step by step, using the
solved exercises under src/. Requires the course-provided `utils/utils.py`
(ServeLLM, display_info, validate_token) and models/data mounted at
/app/models and /app/data, same as the notebook.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from step01_setup import BASE_MODEL, SFT_MODEL, RL_MODEL, LLAMA_GUARD  # noqa: E402
from step02_example_prompts import TEST_PROMPTS, EXPECTED_KEYWORDS  # noqa: E402
from step03_process_prompts import process_prompts  # noqa: E402
from step04_scoring import score_all_responses, build_comparison_table  # noqa: E402
from step05_gsm8k_dataset import load_gsm8k  # noqa: E402
from step07_model_evaluation import evaluate_model_correctness  # noqa: E402
from step08_safety_setup import load_safety_dataset, BENIGN_PROMPTS  # noqa: E402
from step11_safety_pipeline import evaluate_safety_model, analyze_safety_categories  # noqa: E402
from step12_cleanup import cleanup  # noqa: E402


def run_example_prompt_comparison():
    base_results = process_prompts(BASE_MODEL, TEST_PROMPTS)
    sft_results = process_prompts(SFT_MODEL, TEST_PROMPTS)
    rl_results = process_prompts(RL_MODEL, TEST_PROMPTS)

    base_scores, base_avg = score_all_responses(base_results, EXPECTED_KEYWORDS)
    sft_scores, sft_avg = score_all_responses(sft_results, EXPECTED_KEYWORDS)
    rl_scores, rl_avg = score_all_responses(rl_results, EXPECTED_KEYWORDS)

    comparison_df = build_comparison_table(
        TEST_PROMPTS, EXPECTED_KEYWORDS, base_scores, sft_scores, rl_scores
    )
    print(comparison_df.to_string(index=False))
    print(f"\nAverage Scores -> Base: {base_avg:.2f}  SFT: {sft_avg:.2f}  RL: {rl_avg:.2f}")


def run_gsm8k_correctness(num_samples=30):
    gsm8k_dataset = load_gsm8k()
    models_to_test = {"Base": BASE_MODEL, "SFT": SFT_MODEL, "RL": RL_MODEL}

    results = {}
    for name, model_path in models_to_test.items():
        accuracy, details = evaluate_model_correctness(model_path, gsm8k_dataset, num_samples)
        results[name] = accuracy
        print(f"{name} Model Accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%)")
    return results


def run_safety_evaluation(num_harmful=15, num_benign=8):
    safety_dataset = load_safety_dataset()
    evaluation_results = evaluate_safety_model(
        model_path=LLAMA_GUARD,
        harmful_prompts=safety_dataset,
        benign_prompts=BENIGN_PROMPTS,
        num_harmful=num_harmful,
        num_benign=num_benign,
    )
    metrics = evaluation_results["metrics"]
    print(f"Harmful Detection Rate: {metrics['harmful_detection_rate']:.1%}")
    print(f"Benign Acceptance Rate: {metrics['benign_acceptance_rate']:.1%}")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.1%}")
    print(f"False Negative Rate: {metrics['false_negative_rate']:.1%}")

    top_categories = analyze_safety_categories(evaluation_results["harmful_results"])
    print("Most frequent violation categories:", top_categories)
    return evaluation_results


def main():
    run_example_prompt_comparison()
    run_gsm8k_correctness()
    run_safety_evaluation()
    cleanup()


if __name__ == "__main__":
    main()
