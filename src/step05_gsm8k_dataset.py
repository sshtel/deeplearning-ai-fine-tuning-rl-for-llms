"""Step 5: Load the GSM8K dataset used for correctness evaluation."""
from step01_setup import display_info, load_from_disk


def load_gsm8k():
    display_info("Loading GSM8K dataset...")
    return load_from_disk("/app/data/gsm8k", "main")["test"].shuffle(seed=42)


if __name__ == "__main__":
    gsm8k_dataset = load_gsm8k()
    sample = gsm8k_dataset[0]
    print("Example GSM8K problem:")
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
