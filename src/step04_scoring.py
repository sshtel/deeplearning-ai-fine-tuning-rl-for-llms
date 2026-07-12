"""Step 4: Score model responses against expected keywords."""
import pandas as pd


def score_response(response, expected_keyword):
    """Score a single response based on whether it contains the expected keyword."""
    response_lower = response.lower()
    keyword_lower = expected_keyword.lower()
    return 1 if keyword_lower in response_lower else 0


def score_all_responses(model_results, expected_keywords):
    """Score all responses for a model. Returns list of scores and average score."""
    scores = [score_response(r, k) for r, k in zip(model_results, expected_keywords)]
    avg_score = sum(scores) / len(scores)
    return scores, avg_score


def build_comparison_table(prompts, expected_keywords, base_scores, sft_scores, rl_scores):
    return pd.DataFrame({
        "Prompt": [f"Prompt {i + 1}" for i in range(len(prompts))],
        "Expected": expected_keywords,
        "Base Score": base_scores,
        "SFT Score": sft_scores,
        "RL Score": rl_scores,
    })


if __name__ == "__main__":
    from step02_example_prompts import TEST_PROMPTS, EXPECTED_KEYWORDS
    from step03_process_prompts import process_prompts
    from step01_setup import BASE_MODEL, SFT_MODEL, RL_MODEL

    base_model_results = process_prompts(BASE_MODEL, TEST_PROMPTS)
    sft_model_results = process_prompts(SFT_MODEL, TEST_PROMPTS)
    rl_model_results = process_prompts(RL_MODEL, TEST_PROMPTS)

    base_scores, base_avg = score_all_responses(base_model_results, EXPECTED_KEYWORDS)
    sft_scores, sft_avg = score_all_responses(sft_model_results, EXPECTED_KEYWORDS)
    rl_scores, rl_avg = score_all_responses(rl_model_results, EXPECTED_KEYWORDS)

    comparison_df = build_comparison_table(
        TEST_PROMPTS, EXPECTED_KEYWORDS, base_scores, sft_scores, rl_scores
    )

    print("SCORING RESULTS:")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    print("\nAverage Scores:")
    print(f"Base Model: {base_avg:.2f}")
    print(f"SFT Model:  {sft_avg:.2f}")
    print(f"RL Model:   {rl_avg:.2f}")
