"""Measure triage accuracy on labelled tickets with the real triage prompt and model.

    python scripts/measure_routing.py                 # uses data/labelled_tickets.csv
    python scripts/measure_routing.py --file my.csv   # your own labelled tickets

CSV columns: ticket,queue,ticket_type,priority. Needs GOOGLE_API_KEY (see .env.example).
The 21 sample tickets only prove the script works. Judge routing accuracy on a large set
of your own historical tickets, with labels set by your support team.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpdesk_copilot import config  # noqa: E402
from helpdesk_copilot.prompts import TRIAGE_INSTRUCTION  # noqa: E402
from helpdesk_copilot.schemas import Triage  # noqa: E402


def score(rows: list[dict], preds: list[Triage]) -> dict:
    """Pure function: compare labels with predictions."""
    n = len(rows)
    if n == 0 or n != len(preds):
        raise ValueError("rows and predictions must be the same non-zero length")
    fields = ("queue", "ticket_type", "priority")
    correct = {f: sum(getattr(p, f) == r[f] for r, p in zip(rows, preds)) for f in fields}
    all_three = sum(all(getattr(p, f) == r[f] for f in fields) for r, p in zip(rows, preds))
    misses = Counter(
        (r["queue"], p.queue) for r, p in zip(rows, preds) if r["queue"] != p.queue
    )
    return {
        "n": n,
        **{f"{f}_accuracy": correct[f] / n for f in fields},
        "exact_match_accuracy": all_three / n,
        "queue_confusions": dict(misses),
    }


_client = None


def _get_client():
    """One client for all tickets, retrying temporary errors (503, 429) like the app does."""
    global _client
    if _client is None:
        from google import genai
        from google.genai import types

        _client = genai.Client(  # reads GOOGLE_API_KEY from the environment
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    attempts=config.LLM_RETRY_ATTEMPTS, initial_delay=2, max_delay=30
                )
            )
        )
    return _client


def classify(ticket: str) -> Triage:
    from google.genai import types

    client = _get_client()
    resp = client.models.generate_content(
        model=config.ROUTING_MODEL,
        contents=ticket,
        config=types.GenerateContentConfig(
            system_instruction=TRIAGE_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=Triage,
            temperature=0,
        ),
    )
    return Triage.model_validate_json(resp.text)


def classify_with_wait(ticket: str, max_wait: float = 90.0, tries: int = 4) -> Triage:
    """Free-tier models also have a per-minute limit (for example 5 requests a minute).
    When Google says "retry in 23s" and that is short, wait and try again.
    A long wait means the daily quota is gone, so the error is raised."""
    for attempt in range(tries):
        try:
            return classify(ticket)
        except Exception as exc:
            if "RESOURCE_EXHAUSTED" not in str(exc) or attempt == tries - 1:
                raise
            m = re.search(r"retry in ([0-9.]+)s", str(exc))
            wait = float(m.group(1)) + 1 if m else 15.0
            if wait > max_wait:
                raise
            print(f"  per-minute limit, waiting {wait:.0f}s ...", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=str(Path(__file__).resolve().parent.parent / "data" / "labelled_tickets.csv"))
    parser.add_argument("--limit", type=int, default=0, help="score only the first N tickets (saves quota)")
    args = parser.parse_args()

    with open(args.file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    preds: list[Triage] = []
    for r in rows:
        try:
            preds.append(classify_with_wait(r["ticket"]))
        except Exception as exc:  # a daily quota or a closed model: score what finished
            print(f"Stopped after {len(preds)} of {len(rows)} tickets: {str(exc)[:300]}")
            print("Quota is per model and per day on the free tier. Use --limit, set ROUTING_MODEL "
                  "in .env to another model, wait for the reset, or enable billing.\n")
            break
    if not preds:
        sys.exit(1)
    rows = rows[: len(preds)]
    result = score(rows, preds)

    print(f"Tickets: {result['n']}  (model: {config.ROUTING_MODEL})")
    for key in ("queue_accuracy", "ticket_type_accuracy", "priority_accuracy", "exact_match_accuracy"):
        print(f"{key:24s} {result[key]:.1%}")
    if result["queue_confusions"]:
        print("Queue confusions (true -> predicted):", result["queue_confusions"])


if __name__ == "__main__":
    main()
