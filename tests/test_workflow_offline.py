"""Run the real workflow graph end to end with a scripted fake model (no API key)."""

import pytest
from google.adk.runners import InMemoryRunner
from google.genai import types

from helpdesk_copilot.agent import build_workflow
from tests.fakes import ScriptedLlm

TICKET = "Hi, I bought the annual plan 10 days ago and want a refund. Is that possible?"

TRIAGE_OK = {"queue": "billing", "ticket_type": "question", "priority": "P3", "confidence": 0.93, "rationale": "refund question"}
DRAFT_OK = {
    "reply": "Thanks for reaching out. Customers can request a refund within 30 days of the original purchase date.",
    "claims": [{"text": "Refunds can be requested within 30 days of purchase.", "source_ids": ["refund-policy#1"]}],
    "needs_human": False,
}
JUDGE_OK = {"verdicts": [{"claim_index": 1, "supported": True, "note": "stated in the policy"}]}


async def run_ticket(triage, draft=None, judge=None, ticket=TICKET):
    models = {
        "triage": ScriptedLlm(script=[triage], prompts=[]),
        "draft": ScriptedLlm(script=[draft] if draft is not None else [], prompts=[]),
        "judge": ScriptedLlm(script=[judge] if judge is not None else [], prompts=[]),
    }
    wf = build_workflow(models["triage"], models["draft"], models["judge"])
    runner = InMemoryRunner(agent=wf, app_name="test")
    session = await runner.session_service.create_session(app_name="test", user_id="u")
    events = []
    async for ev in runner.run_async(
        user_id="u", session_id=session.id, new_message=types.Content(role="user", parts=[types.Part(text=ticket)])
    ):
        events.append(ev)
    final = [e for e in events if isinstance(e.output, dict) and "decision" in e.output]
    assert len(final) == 1, f"expected exactly one final decision event, got {len(final)}"
    return final[0].output, models


async def test_happy_path_delivers_with_trail():
    out, models = await run_ticket(TRIAGE_OK, DRAFT_OK, JUDGE_OK)
    assert out["decision"] == "auto_reply"
    assert "30 days" in out["reply"]
    assert [s["step"] for s in out["trail"]] == ["intake", "retrieve", "triage", "draft", "validate", "deliver"]
    # The draft agent really received the retrieved passages and the triage summary.
    assert "[refund-policy#1]" in models["draft"].prompts[0]
    assert "queue=billing" in models["draft"].prompts[0]


async def test_unsupported_claim_escalates():
    bad_judge = {"verdicts": [{"claim_index": 1, "supported": False, "note": "not in the passages"}]}
    out, _ = await run_ticket(TRIAGE_OK, DRAFT_OK, bad_judge)
    assert out["decision"] == "escalate_to_human"
    assert any("not supported" in r for r in out["reasons"])
    assert out["unsent_draft"]


async def test_invented_number_escalates():
    draft = {**DRAFT_OK, "reply": "You can request a refund within 90 days of purchase."}
    out, _ = await run_ticket(TRIAGE_OK, draft, JUDGE_OK)
    assert out["decision"] == "escalate_to_human"
    assert any("90" in r for r in out["reasons"])


async def test_low_confidence_skips_drafting():
    out, models = await run_ticket({**TRIAGE_OK, "confidence": 0.4})
    assert out["decision"] == "escalate_to_human"
    assert any("low triage confidence" in r for r in out["reasons"])
    assert models["draft"].prompts == []  # the draft agent never ran


async def test_security_ticket_goes_straight_to_a_human():
    triage = {"queue": "security", "ticket_type": "incident", "priority": "P1", "confidence": 0.98, "rationale": "compromise"}
    out, models = await run_ticket(triage, ticket="Someone logged in to my account from another country.")
    assert out["decision"] == "escalate_to_human"
    assert any("security" in r for r in out["reasons"])
    assert models["draft"].prompts == []


async def test_draft_without_claims_skips_judge_and_escalates():
    draft = {"reply": "I am not sure.", "claims": [], "needs_human": True}
    out, models = await run_ticket(TRIAGE_OK, draft)
    assert out["decision"] == "escalate_to_human"
    assert models["judge"].prompts == []
