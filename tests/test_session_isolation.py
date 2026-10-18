"""Two tickets in the SAME session (like the adk web chat window).

Per-ticket fields must not leak from one ticket into the next.
"""

from google.adk.runners import InMemoryRunner
from google.genai import types

from helpdesk_copilot.agent import build_workflow
from tests.fakes import ScriptedLlm
from tests.test_workflow_offline import DRAFT_OK, JUDGE_OK, TRIAGE_OK

LOW_CONFIDENCE = {**TRIAGE_OK, "queue": "technical", "ticket_type": "incident", "priority": "P2", "confidence": 0.35}


async def _send(runner, session_id, text):
    events = []
    async for ev in runner.run_async(
        user_id="u", session_id=session_id, new_message=types.Content(role="user", parts=[types.Part(text=text)])
    ):
        events.append(ev)
    final = [e for e in events if isinstance(e.output, dict) and "decision" in e.output]
    assert len(final) == 1
    return final[0].output


async def test_second_ticket_does_not_inherit_first_tickets_draft():
    triage = ScriptedLlm(script=[TRIAGE_OK, LOW_CONFIDENCE], prompts=[])
    draft = ScriptedLlm(script=[DRAFT_OK], prompts=[])  # only the first ticket may be drafted
    judge = ScriptedLlm(script=[JUDGE_OK], prompts=[])
    runner = InMemoryRunner(agent=build_workflow(triage, draft, judge), app_name="test")
    session = await runner.session_service.create_session(app_name="test", user_id="u")

    first = await _send(runner, session.id, "I bought the annual plan 10 days ago and want a refund. Is that possible?")
    assert first["decision"] == "auto_reply"

    second = await _send(runner, session.id, "it is not working, please fix")
    assert second["decision"] == "escalate_to_human"
    assert second["unsent_draft"] is None  # no draft was written for this ticket
    assert second["reasons"] == ["low triage confidence 0.35 < 0.70"]
    assert [s["step"] for s in second["trail"]] == ["intake", "retrieve", "triage", "escalate"]
