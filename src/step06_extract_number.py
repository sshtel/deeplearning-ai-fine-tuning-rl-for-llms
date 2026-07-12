"""Step 6 (Exercise 2): Extract the final numerical answer from model output."""
import re

GSM8K_ANSWER_RE = r"####\s*([-+]?\d+(?:\.\d+)?)"
LAST_NUMBER_RE = r"[-+]?\d+(?:\.\d+)?"


def extract_number(text):
    """
    Extract the final numerical answer from a model's generated output.
    GSM8K answers are formatted like '#### 42', but we also fall back to the
    last number in the text.
    """
    gsm8k_format = re.search(GSM8K_ANSWER_RE, text)
    if gsm8k_format:
        try:
            return float(gsm8k_format.group(1))
        except ValueError:
            pass

    numbers = re.findall(LAST_NUMBER_RE, text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


if __name__ == "__main__":
    assert extract_number("We calculate it as 6 * 7 = 42\n#### 42") == 42.0
    assert extract_number("The answer is #### -12.5") == -12.5
    assert extract_number("Add 1 and 2 to get 3.") == 3.0
    assert extract_number("No numbers at all.") is None
    print("All extract_number tests passed.")
