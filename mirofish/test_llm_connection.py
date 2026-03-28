"""
mirofish/test_llm_connection.py

Quick smoke-test: sends a simple message through the configured LLM endpoint
and prints the response.  Run from the mirofish/ directory:

    uv run python test_llm_connection.py
    # or after activating .venv:
    python test_llm_connection.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from mirofish/ root
load_dotenv(Path(__file__).parent / ".env")

API_KEY  = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
MODEL    = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
RESET = "\033[0m"

def main():
    print(f"\n{BOLD}MiroFish LLM Connection Test{RESET}")
    print("=" * 48)
    print(f"  Base URL : {BASE_URL}")
    print(f"  Model    : {MODEL}")
    key_preview = (API_KEY[:8] + "..." + API_KEY[-4:]) if len(API_KEY) > 12 else "NOT SET"
    print(f"  API Key  : {key_preview}")
    print("=" * 48)

    if not API_KEY or API_KEY.startswith("your_"):
        print(f"\n{RED}[FAIL]{RESET} LLM_API_KEY is not set in mirofish/.env")
        print("  Edit mirofish/.env and add your Anthropic API key.")
        sys.exit(1)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

        print(f"\nSending test message to {MODEL}...")
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=120,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a NEPSE market analyst assistant. "
                        "Reply in exactly 2 sentences: confirm you are Claude, "
                        "and state one fact about Nepal's stock market."
                    ),
                }
            ],
        )

        reply = response.choices[0].message.content.strip()
        print(f"\n{GREEN}[PASS]{RESET} LLM responded successfully.\n")
        print(f"{CYAN}Response:{RESET}")
        print(f"  {reply}\n")

        # Metadata
        usage = response.usage
        print(f"  Model       : {response.model}")
        print(f"  Prompt tok  : {usage.prompt_tokens}")
        print(f"  Completion  : {usage.completion_tokens}")
        print(f"  Total tok   : {usage.total_tokens}")
        print(f"\n{GREEN}Claude is responding correctly. MiroFish LLM backend is ready.{RESET}\n")

    except Exception as exc:
        print(f"\n{RED}[FAIL]{RESET} LLM call failed: {exc}")
        print("\nTroubleshooting:")
        print("  1. Check LLM_API_KEY is a valid Anthropic key (sk-ant-...)")
        print("  2. Check LLM_BASE_URL = https://api.anthropic.com/v1")
        print("  3. Check LLM_MODEL_NAME = claude-sonnet-4-5")
        sys.exit(1)


if __name__ == "__main__":
    main()
