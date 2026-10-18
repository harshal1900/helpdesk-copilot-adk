"""Central configuration. Everything comes from environment variables / .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Load a .env from the current folder or any parent (works for scripts and tests).
# `adk web` / `adk run` also load .env on their own, and do not override variables
# that are already set in your shell.
load_dotenv(find_dotenv(usecwd=True))

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = Path(os.getenv("KB_DIR", ROOT / "helpdesk_copilot" / "kb"))
CACHE_DIR = ROOT / ".cache"

TRIAGE_MODEL = os.getenv("TRIAGE_MODEL", "gemini-3.5-flash")
DRAFT_MODEL = os.getenv("DRAFT_MODEL", "gemini-3.5-flash")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.5-flash")
# Models used only by the evaluation scripts. Quotas are counted per model, so a
# separate model keeps evals from using up the app's daily quota.
EVAL_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", JUDGE_MODEL)  # judge inside `adk eval`
ROUTING_MODEL = os.getenv("ROUTING_MODEL", TRIAGE_MODEL)  # scripts/measure_routing.py

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "gemini")  # gemini | local
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
TOP_K = int(os.getenv("TOP_K", "4"))

MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.45"))
MIN_TRIAGE_CONFIDENCE = float(os.getenv("MIN_TRIAGE_CONFIDENCE", "0.70"))

AUDIT_DIR = os.getenv("AUDIT_DIR")  # optional; None means do not write audit files

# Total attempts per Gemini call (first try included) when Google returns a temporary
# error such as 503 "high demand" or 429 "rate limit". Waits grow between attempts.
LLM_RETRY_ATTEMPTS = int(os.getenv("LLM_RETRY_ATTEMPTS", "5"))
