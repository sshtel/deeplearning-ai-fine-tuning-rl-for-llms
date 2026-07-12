"""Step 9 (Exercise 4): Parse Llama Guard's structured safety classification output."""
import re


def parse_llama_guard_response(output: str):
    """
    Parse Llama Guard model responses into structured format.

    Llama Guard outputs either:
    - "safe" for acceptable content
    - "unsafe" followed by violated category codes on new lines (e.g., "unsafe\\nS1\\nS5")

    Returns:
        dict: {
            'classification': 'safe' | 'unsafe' | 'unknown',
            'categories': list of violated categories (e.g., ['S1', 'S5'])
        }
    """
    if not isinstance(output, str) or not output.strip():
        return {"classification": "unknown", "categories": []}

    text = output.strip().lower()

    if "unsafe" in text:
        categories = re.findall(r"s\d+", text)
        return {
            "classification": "unsafe",
            "categories": [c.upper() for c in categories],
        }
    elif "safe" in text:
        return {
            "classification": "safe",
            "categories": [],
        }
    else:
        return {
            "classification": "unknown",
            "categories": [],
        }


if __name__ == "__main__":
    print("Testing parse_llama_guard_response function:")
    test_responses = [
        "unsafe\nS1\nS5",   # Multiple violations
        "safe",              # Safe content
        "unsafe\nS2",        # Single violation
        "This is invalid",  # Invalid response
        "",                  # Empty response
    ]

    for response in test_responses:
        result = parse_llama_guard_response(response)
        print(f"Input: {response!r}")
        print(f"Output: {result}")
        print()
