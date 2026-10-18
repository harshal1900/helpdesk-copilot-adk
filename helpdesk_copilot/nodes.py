"""Deterministic workflow nodes (plain Python functions, no LLM).

Every node appends a step to state["trail"]. That list is the audit trail.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from google.adk import Context, Event

from . import config
from .retrieval import get_kb
from .schemas import Draft, Judgement, Triage
from .validation import check_response


# --------------------------------------------------------------------------- helpers
def _as_text(value: Any) -> str:
    """The first node receives the user's message. Accept a string or a Content object."""
    if isinstance(value, str):
        return value
    parts = getattr(value, "parts", None)
    if parts:
        return "".join(getattr(p, "text", "") or "" for p in parts)
    return str(value)


def _coerce(model, value: Any):
    """Agent outputs arrive as a model instance, a dict or a JSON string. Normalise."""
    if isinstance(value, model):
        return value
    if isinstance(value, (str, bytes)):
        return model.model_validate_json(value)
    return model.model_validate(value)


def _try_coerce(model, value: Any):
    try:
        return _coerce(model, value)
    except Exception:
        return None


def _trail(ctx: Context, step: str, **detail: Any) -> list[dict]:
    """Return the current trail plus one new step (state is updated through Event(state=...))."""
    trail = list(ctx.state.get("trail", []))
    trail.append({"step": step, "t": round(time.time(), 3), **detail})
    return trail


def _write_audit(payload: dict) -> None:
    if not config.AUDIT_DIR:
        return
    folder = Path(config.AUDIT_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}.json"
    (folder / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- nodes
def intake(node_input: Any, ctx: Context):
    ticket = _as_text(node_input).strip()
    return Event(
        output=ticket,
        state={"ticket": ticket, "trail": [{"step": "intake", "t": round(time.time(), 3), "chars": len(ticket)}]},
    )


def retrieve(node_input: str, ctx: Context):
    ticket = ctx.state.get("ticket", node_input)
    found = get_kb().search(ticket, k=config.TOP_K)
    passages = [p.to_dict() for p in found]
    passages_text = "\n".join(f"[{p.id}] {p.text}" for p in found)
    return Event(
        output=ticket,
        state={
            "passages": passages,
            "passages_text": passages_text,
            "trail": _trail(ctx, "retrieve", hits=[{"id": p["id"], "score": p["score"]} for p in passages]),
        },
    )


def route_triage(node_input: Any, ctx: Context):
    """Decide, in plain code, whether the model may continue on its own."""
    triage = _coerce(Triage, node_input)
    reasons: list[str] = []
    if triage.queue == "security":
        reasons.append("security queue: must go to the Security team")
    if triage.priority == "P1":
        reasons.append("priority P1: needs a human immediately")
    if triage.confidence < config.MIN_TRIAGE_CONFIDENCE:
        reasons.append(f"low triage confidence {triage.confidence:.2f} < {config.MIN_TRIAGE_CONFIDENCE:.2f}")

    summary = f"queue={triage.queue}, type={triage.ticket_type}, priority={triage.priority}"
    route = "escalate" if reasons else "draft"
    return Event(
        output=ctx.state.get("ticket", ""),
        route=route,
        state={
            "triage": triage.model_dump(),
            "triage_summary": summary,
            "reasons": reasons,
            "trail": _trail(ctx, "triage", **triage.model_dump(), route=route),
        },
    )


def save_draft(node_input: Any, ctx: Context):
    draft = _coerce(Draft, node_input)
    claims_text = "\n".join(
        f"{i}. {c.text} [sources: {', '.join(c.source_ids) or 'none'}]"
        for i, c in enumerate(draft.claims, start=1)
    )
    return Event(
        output=claims_text,
        route="judge" if draft.claims else "skip",
        state={
            "draft": draft.model_dump(),
            "trail": _trail(ctx, "draft", claims=len(draft.claims), needs_human=draft.needs_human),
        },
    )


def validate(node_input: Any, ctx: Context):
    draft = Draft.model_validate(ctx.state.get("draft"))
    judgement = _try_coerce(Judgement, node_input)
    reasons = check_response(
        draft=draft,
        judgement=judgement,
        passages=list(ctx.state.get("passages", [])),
        ticket=ctx.state.get("ticket", ""),
        min_retrieval_score=config.MIN_RETRIEVAL_SCORE,
    )
    route = "escalate" if reasons else "deliver"
    return Event(
        output=draft.reply,
        route=route,
        state={
            "reasons": reasons,
            "trail": _trail(ctx, "validate", passed=not reasons, reasons=reasons, route=route),
        },
    )


def deliver(node_input: Any, ctx: Context):
    draft = ctx.state.get("draft", {})
    payload = {
        "decision": "auto_reply",
        "reply": draft.get("reply", ""),
        "triage": ctx.state.get("triage"),
        "evidence": [p["id"] for p in ctx.state.get("passages", [])],
        "trail": _trail(ctx, "deliver"),
    }
    _write_audit(payload)
    sources = sorted({sid for c in draft.get("claims", []) for sid in c.get("source_ids", [])})
    message = (
        f"DECISION: auto-reply approved\n"
        f"Triage: {ctx.state.get('triage_summary')}\n"
        f"Sources: {', '.join(sources)}\n\n"
        f"{payload['reply']}"
    )
    return Event(message=message, output=payload)


def escalate(node_input: Any, ctx: Context):
    draft = ctx.state.get("draft")
    payload = {
        "decision": "escalate_to_human",
        "reasons": list(ctx.state.get("reasons", [])),
        "triage": ctx.state.get("triage"),
        "evidence": [p["id"] for p in ctx.state.get("passages", [])],
        "unsent_draft": draft.get("reply") if draft else None,
        "trail": _trail(ctx, "escalate"),
    }
    _write_audit(payload)
    reasons = "\n".join(f"- {r}" for r in payload["reasons"]) or "- (none recorded)"
    message = (
        "DECISION: escalated to a human agent\n"
        f"Triage: {ctx.state.get('triage_summary', 'not available')}\n"
        f"Reasons:\n{reasons}"
    )
    return Event(message=message, output=payload)
