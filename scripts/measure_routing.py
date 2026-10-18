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
import sys
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


def classify(ticket: str) -> Triage:
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GOOGLE_API_KEY from the environment
    resp = client.models.generate_content(
        model=config.TRIAGE_MODEL,
        contents=ticket,
        config=types.GenerateContentConfig(
            system_instruction=TRIAGE_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=Triage,
            temperature=0,
        ),
    )
    return Triage.model_validate_json(resp.text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=str(Path(__file__).resolve().parent.parent / "data" / "labelled_tickets.csv"))
    args = parser.parse_args()

    with open(args.file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    preds = [classify(r["ticket"]) for r in rows]
    result = score(rows, preds)

    print(f"Tickets: {result['n']}  (model: {config.TRIAGE_MODEL})")
    for key in ("queue_accuracy", "ticket_type_accuracy", "priority_accuracy", "exact_match_accuracy"):
        print(f"{key:24s} {result[key]:.1%}")
    if result["queue_confusions"]:
        print("Queue confusions (true -> predicted):", result["queue_confusions"])


if __name__ == "__main__":
    main()
