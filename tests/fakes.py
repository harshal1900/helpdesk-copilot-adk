"""A scripted fake model, so the whole workflow can run in tests without calling Gemini."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models._capabilities import LlmCapabilities
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types


class ScriptedLlm(BaseLlm):
    """Returns the next scripted JSON answer each time it is called, and records the prompts."""

    model: str = "scripted"
    script: list[Any] = []
    prompts: list[str] = []

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=True)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        system = ""
        if llm_request.config and llm_request.config.system_instruction:
            system = str(llm_request.config.system_instruction)
        user = " ".join(
            p.text or "" for c in llm_request.contents for p in (c.parts or [])
        )
        self.prompts.append(system + "\n" + user)
        answer = self.script.pop(0)
        text = answer if isinstance(answer, str) else json.dumps(answer)
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))
