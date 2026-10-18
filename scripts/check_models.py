"""List the Gemini models your API key can call, so you can pick valid names for .env.

    python scripts/check_models.py            # models that support generateContent
    python scripts/check_models.py flash      # only names containing "flash"

This is a free metadata call and does not use your generation quota. Free-tier
daily limits are per model and per project: see https://aistudio.google.com/rate-limit
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpdesk_copilot import config  # noqa: E402,F401  (loads .env)


def main() -> None:
    from google import genai

    needle = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    client = genai.Client()
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions and needle in m.name.lower():
            names.append(m.name.removeprefix("models/"))
    for name in sorted(names):
        print(name)
    print(f"\n{len(names)} models. Put chosen names into TRIAGE_MODEL, DRAFT_MODEL, JUDGE_MODEL, EVAL_JUDGE_MODEL in .env")


if __name__ == "__main__":
    main()
