"""Step 12: Release GPU memory used by the models."""
from step01_setup import ServeLLM


def cleanup():
    ServeLLM.cleanup_all()
    print("Lab completed! GPU memory cleaned up.")


if __name__ == "__main__":
    cleanup()
