"""Deterministic checks that run on every drafted reply. Plain Python, no LLM.

check_response() returns a list of human-readable reasons. An empty list means the
reply passed every check and may be delivered. Any reason means: escalate to a human.
"""

from __future__ import annotations

import re

from .schemas import Draft, Judgement

# Phrases a first response must never contain (promises the agent cannot make).
BANNED_PATTERNS: dict[str, str] = {
    "guarantee": r"\bguarantee[sd]?\b",
    "refund promise": r"\b(will be refunded|refund (has been|is) approved|i have refunded|we will refund)\b",
    "legal advice": r"\blegal advice\b",
}

_NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")


def _numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in _NUMBER.finditer(text)}


def check_response(
    draft: Draft,
    judgement: Judgement | None,
    passages: list[dict],
    ticket: str,
    min_retrieval_score: float,
) -> list[str]:
    reasons: list[str] = []
    by_id = {p["id"]: p for p in passages}

    # 1. Is there enough evidence at all?
    best = max((p["score"] for p in passages), default=0.0)
    if not passages or best < min_retrieval_score:
        reasons.append(f"weak retrieval: best passage score {best:.2f} < {min_retrieval_score:.2f}")

    # 2. Did the drafting step itself say it cannot answer?
    if draft.needs_human:
        reasons.append("draft agent flagged that the knowledge base cannot answer this")

    # 3. Every reply must make at least one cited claim.
    if not draft.claims:
        reasons.append("reply has no cited claims")

    # 4. Every claim must cite passages that were actually retrieved.
    cited_text = ""
    for i, claim in enumerate(draft.claims, start=1):
        if not claim.source_ids:
            reasons.append(f"claim {i} has no source")
        for sid in claim.source_ids:
            if sid not in by_id:
                reasons.append(f"claim {i} cites unknown source {sid}")
            else:
                cited_text += " " + by_id[sid]["text"]

    # 5. A second model must confirm each claim is supported by the passages.
    if draft.claims:
        if judgement is None:
            reasons.append("no grounding judgement available")
        else:
            verdicts = {v.claim_index: v for v in judgement.verdicts}
            for i in range(1, len(draft.claims) + 1):
                v = verdicts.get(i)
                if v is None:
                    reasons.append(f"claim {i} was not judged")
                elif not v.supported:
                    reasons.append(f"claim {i} not supported: {v.note}")

    # 6. Policy rules.
    for name, pattern in BANNED_PATTERNS.items():
        if re.search(pattern, draft.reply, flags=re.IGNORECASE):
            reasons.append(f"policy: reply contains banned content ({name})")

    # 7. Numbers (amounts, days, limits) must appear in a cited passage or the ticket.
    allowed = _numbers(cited_text) | _numbers(ticket)
    unknown = sorted(n for n in _numbers(draft.reply) if n not in allowed)
    if unknown:
        reasons.append(f"policy: numbers not found in cited passages: {', '.join(unknown)}")

    return reasons
