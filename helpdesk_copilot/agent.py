"""The Helpdesk Copilot workflow.

ticket -> intake -> retrieve -> triage (LLM) -> route_triage
            route "draft"    -> draft (LLM) -> save_draft -> judge (LLM) -> validate
            route "escalate" -> escalate
validate: route "deliver" -> deliver, route "escalate" -> escalate

LLMs are used only where language understanding is needed (triage, draft, judge).
Routing, validation and escalation are ordinary Python.
"""

from __future__ import annotations

from google.adk import Agent, Workflow
from google.adk.models import Gemini
from google.genai import types

from . import config
from .nodes import deliver, escalate, intake, retrieve, route_triage, save_draft, validate
from .prompts import DRAFT_INSTRUCTION, JUDGE_INSTRUCTION, TRIAGE_INSTRUCTION
from .schemas import Draft, Judgement, Triage


def _model(model):
    """A model name becomes a Gemini model that retries temporary errors (503, 429)."""
    if not isinstance(model, str):
        return model  # already a model object, for example the fake model in tests
    return Gemini(
        model=model,
        retry_options=types.HttpRetryOptions(
            attempts=config.LLM_RETRY_ATTEMPTS, initial_delay=2, max_delay=30
        ),
    )


def build_workflow(triage_model=None, draft_model=None, judge_model=None) -> Workflow:
    """Build the graph. Models can be names or model objects (tests pass a fake model)."""
    triage_agent = Agent(
        name="triage",
        model=_model(triage_model or config.TRIAGE_MODEL),
        mode="single_turn",
        output_schema=Triage,
        instruction=TRIAGE_INSTRUCTION,
    )
    draft_agent = Agent(
        name="draft",
        model=_model(draft_model or config.DRAFT_MODEL),
        mode="single_turn",
        output_schema=Draft,
        instruction=DRAFT_INSTRUCTION,
    )
    judge_agent = Agent(
        name="judge",
        model=_model(judge_model or config.JUDGE_MODEL),
        mode="single_turn",
        output_schema=Judgement,
        instruction=JUDGE_INSTRUCTION,
    )

    return Workflow(
        name="helpdesk_copilot",
        edges=[
            ("START", intake, retrieve, triage_agent, route_triage),
            (route_triage, {"draft": draft_agent, "escalate": escalate}),
            (draft_agent, save_draft),
            (save_draft, {"judge": judge_agent, "skip": validate}),
            (judge_agent, validate),
            (validate, {"deliver": deliver, "escalate": escalate}),
        ],
    )


root_agent = build_workflow()
