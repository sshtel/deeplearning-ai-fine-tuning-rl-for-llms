"""Step 3 (Exercise 1): Generate responses from a model for a list of prompts."""
from step01_setup import ServeLLM, BASE_MODEL, SFT_MODEL, RL_MODEL
from step02_example_prompts import TEST_PROMPTS


def process_prompts(model_name, prompts):
    """Process a list of prompts with a given model and return responses."""
    results = []
    with ServeLLM(model_name) as llm:
        for prompt in prompts:
            response = llm.generate_response(prompt)
            results.append(response)
    return results


def _print_results(label, prompts, responses):
    for prompt, response in zip(prompts, responses):
        print(f"\nPrompt: {prompt}")
        shown = response[:200] + "..." if len(response) > 200 else response
        print(f"{label} Response: {shown}")


if __name__ == "__main__":
    print("=" * 50)
    print("PROCESSING BASE MODEL")
    print("=" * 50)
    base_model_results = process_prompts(BASE_MODEL, TEST_PROMPTS)
    _print_results("Base Model", TEST_PROMPTS, base_model_results)

    print("=" * 50)
    print("PROCESSING FINE-TUNED MODEL")
    print("=" * 50)
    sft_model_results = process_prompts(SFT_MODEL, TEST_PROMPTS)
    _print_results("Fine-Tuned Model", TEST_PROMPTS, sft_model_results)

    print("=" * 50)
    print("PROCESSING RL MODEL")
    print("=" * 50)
    rl_model_results = process_prompts(RL_MODEL, TEST_PROMPTS)
    _print_results("RL Model", TEST_PROMPTS, rl_model_results)
