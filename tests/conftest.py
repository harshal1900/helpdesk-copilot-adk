"""Tests run fully offline: no API key, no network. Env vars are set before the app imports."""

import os
import sys
from pathlib import Path

os.environ["EMBEDDING_BACKEND"] = "local"
os.environ["MIN_RETRIEVAL_SCORE"] = "0.15"  # local hashing scores are lower than Gemini scores
os.environ["MIN_TRIAGE_CONFIDENCE"] = "0.70"
os.environ.pop("AUDIT_DIR", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
