"""Step 2: Example prompts used to spot-check model responses."""

# Test prompts for model comparison
TEST_PROMPTS = [
    "What is the area of a rectangle with a length of 8 units and a width of 5 units?",
    "Solve: 2x + 3 = 7",
    "What is the derivative of sin(x)?",
]

# Expected key information in correct answers
EXPECTED_KEYWORDS = [
    "40",      # 8 * 5 = 40
    "x = 2",   # 2x + 3 = 7 -> 2x = 4 -> x = 2
    "cos(x)",  # derivative of sin(x) is cos(x)
]

if __name__ == "__main__":
    print("Test prompts defined:")
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"{i + 1}. {prompt}")
