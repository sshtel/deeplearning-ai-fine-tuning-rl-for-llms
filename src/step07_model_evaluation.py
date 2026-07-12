"""Step 7 (Exercise 3): Evaluate a model's correctness on GSM8K problems."""
from tqdm import tqdm

from step01_setup import ServeLLM
from step06_extract_number import extract_number


def evaluate_model_correctness(model_path, gsm8k_dataset, num_samples=30):
    """
    Evaluate a model's correctness on GSM8K problems.

    Args:
        model_path: Path to the model
        gsm8k_dataset: GSM8K test split (from step05_gsm8k_dataset.load_gsm8k)
        num_samples: Number of samples to test (reduced for faster execution)

    Returns:
        accuracy: Fraction of correct answers
        results: Per-sample details
    """
    print(f"Evaluating {model_path} on {num_samples} GSM8K problems...")

    test_data = gsm8k_dataset.select(range(num_samples))

    correct = 0
    results = []

    with ServeLLM(model_path) as llm:
        for i, sample in enumerate(tqdm(test_data, desc="Processing")):
            prompt = f"Solve this math problem step by step:\n{sample['question']}\n\nAnswer:"

            response = llm.generate_response(prompt, max_tokens=512)

            model_answer = extract_number(response)
            gold_answer = extract_number(sample["answer"])

            is_correct = (
                model_answer is not None
                and gold_answer is not None
                and model_answer == gold_answer
            )

            if is_correct:
                correct += 1

            results.append({
                "question": sample["question"],
                "gold_answer": gold_answer,
                "model_answer": model_answer,
                "correct": is_correct,
            })

            if i < 3:
                print(f"\nExample {i + 1}:")
                print(f"Question: {sample['question'][:100]}...")
                print(f"Gold: {gold_answer}, Model: {model_answer}, Correct: {is_correct}")

    accuracy = correct / num_samples
    return accuracy, results


if __name__ == "__main__":
    from step01_setup import BASE_MODEL, SFT_MODEL, RL_MODEL
    from step05_gsm8k_dataset import load_gsm8k

    gsm8k_dataset = load_gsm8k()

    print("Testing correctness evaluation with fine-tuned model:")
    sft_accuracy, sft_results = evaluate_model_correctness(SFT_MODEL, gsm8k_dataset, num_samples=10)
    print(f"Fine-Tuned Model Accuracy: {sft_accuracy:.2f} ({sft_accuracy * 100:.1f}%)")

    print("Evaluating all three models on correctness...")
    print("This may take several minutes...")

    models_to_test = {"Base": BASE_MODEL, "SFT": SFT_MODEL, "RL": RL_MODEL}
    correctness_results = {}
    num_samples = 30

    for name, model_path in models_to_test.items():
        print(f"\n{'=' * 20} {name.upper()} MODEL {'=' * 20}")
        accuracy, detailed_results = evaluate_model_correctness(model_path, gsm8k_dataset, num_samples)
        correctness_results[name] = {"accuracy": accuracy, "details": detailed_results}
        print(f"{name} Model Accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%)")

    print("\nCORRECTNESS SUMMARY:")
    print("=" * 40)
    for name, results in correctness_results.items():
        print(f"{name:>8} Model: {results['accuracy']:.3f} ({results['accuracy'] * 100:.1f}%)")
