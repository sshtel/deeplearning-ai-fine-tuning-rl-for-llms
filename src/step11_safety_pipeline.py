"""Step 11: Full safety evaluation pipeline - category analysis and end-to-end run."""
from step01_setup import ServeLLM
from step09_parse_llama_guard import parse_llama_guard_response
from step10_safety_metrics import calculate_safety_metrics


def analyze_safety_categories(results):
    """
    Count and rank safety category violations.

    Args:
        results: List of dicts with 'categories' key containing violation lists

    Returns:
        List of tuples: (category_code, count) sorted by frequency (descending)
    """
    category_counts = {}
    for result in results:
        for category in result["categories"]:
            category_counts[category] = category_counts.get(category, 0) + 1

    return sorted(category_counts.items(), key=lambda x: x[1], reverse=True)


def evaluate_safety_model(model_path, harmful_prompts, benign_prompts, num_harmful=10, num_benign=5):
    """
    Comprehensive evaluation of a Llama Guard model on safety classification.

    Args:
        model_path: HuggingFace model path for Llama Guard model
        harmful_prompts: Dataset of harmful prompts to test
        benign_prompts: List of benign prompts for comparison
        num_harmful: Number of harmful prompts to evaluate
        num_benign: Number of benign prompts to evaluate

    Returns:
        dict: Complete evaluation results with metrics and detailed outputs
    """
    print(f"Evaluating safety model: {model_path}")
    print(f"Testing {num_harmful} harmful + {num_benign} benign prompts...")

    harmful_sample = harmful_prompts.select(range(num_harmful))
    benign_sample = benign_prompts[:num_benign]

    harmful_results = []
    benign_results = []

    with ServeLLM(model_path) as llm:

        print("\n--- Testing Harmful Prompts ---")
        for i, sample in enumerate(harmful_sample):
            prompt = sample["Goal"]
            response = llm.generate_response(prompt, max_tokens=64, temperature=0.1)
            parsed = parse_llama_guard_response(response)

            result = {
                "prompt": prompt,
                "response": response,
                "classification": parsed["classification"],
                "categories": parsed["categories"],
            }
            harmful_results.append(result)

            if i == 0:
                print(f"Example harmful prompt: {prompt[:60]}...")
                print(f"Model classification: {parsed['classification']}")
                if parsed["categories"]:
                    print(f"Violation categories: {parsed['categories']}")

        print("\n--- Testing Benign Prompts ---")
        for i, prompt in enumerate(benign_sample):
            response = llm.generate_response(prompt, max_tokens=64, temperature=0.1)
            parsed = parse_llama_guard_response(response)

            result = {
                "prompt": prompt,
                "response": response,
                "classification": parsed["classification"],
                "categories": parsed["categories"],
            }
            benign_results.append(result)

            if i == 0:
                print(f"Example benign prompt: {prompt}")
                print(f"Model classification: {parsed['classification']}")

    metrics = calculate_safety_metrics(harmful_results, benign_results)

    return {
        "harmful_results": harmful_results,
        "benign_results": benign_results,
        "metrics": metrics,
    }


if __name__ == "__main__":
    from step01_setup import LLAMA_GUARD
    from step08_safety_setup import load_safety_dataset, BENIGN_PROMPTS

    safety_dataset = load_safety_dataset()

    print("=" * 60)
    print("RUNNING COMPREHENSIVE SAFETY EVALUATION")
    print("=" * 60)
    print("This model is specifically trained for safety classification tasks")

    evaluation_results = evaluate_safety_model(
        model_path=LLAMA_GUARD,
        harmful_prompts=safety_dataset,
        benign_prompts=BENIGN_PROMPTS,
        num_harmful=15,
        num_benign=8,
    )

    print(f"\n{'=' * 40}")
    print("SAFETY EVALUATION RESULTS")
    print(f"{'=' * 40}")

    metrics = evaluation_results["metrics"]
    print(f"Harmful Detection Rate: {metrics['harmful_detection_rate']:.1%}")
    print(f"Benign Acceptance Rate: {metrics['benign_acceptance_rate']:.1%}")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.1%}")
    print(f"False Negative Rate: {metrics['false_negative_rate']:.1%}")

    print("\nInterpretation:")
    print(f"- The model correctly identified {metrics['harmful_detection_rate']:.1%} of harmful content")
    print(f"- The model correctly accepted {metrics['benign_acceptance_rate']:.1%} of benign content")

    top_categories = analyze_safety_categories(evaluation_results["harmful_results"])
    print("\nMost frequent violation categories:", top_categories)
