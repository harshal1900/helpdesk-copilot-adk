"""Structured outputs. Each LLM step must return one of these shapes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Queue = Literal["billing", "technical", "account", "security", "general"]
TicketType = Literal["question", "incident", "request"]
Priority = Literal["P1", "P2", "P3", "P4"]


class Triage(BaseModel):
    queue: Queue = Field(description="Which team queue should own this ticket.")
    ticket_type: TicketType = Field(description="question, incident or request.")
    priority: Priority = Field(description="P1 is most urgent, P4 is least.")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Your confidence in this classification, 0 to 1."
    )
    rationale: str = Field(description="One short sentence explaining the choice.")


class Claim(BaseModel):
    text: str = Field(description="One factual statement made in the reply.")
    source_ids: list[str] = Field(
        description="IDs of the passages that support this statement, for example refund-policy#1."
    )


class Draft(BaseModel):
    reply: str = Field(description="The customer-facing first response.")
    claims: list[Claim] = Field(
        description="Every factual statement in the reply, each with its supporting passage IDs."
    )
    needs_human: bool = Field(
        default=False,
        description="True if the passages do not contain enough information to answer.",
    )


class ClaimVerdict(BaseModel):
    claim_index: int = Field(description="1-based number of the claim being judged.")
    supported: bool = Field(description="True only if the passages clearly support the claim.")
    note: str = Field(description="Short reason.")


class Judgement(BaseModel):
    verdicts: list[ClaimVerdict]
