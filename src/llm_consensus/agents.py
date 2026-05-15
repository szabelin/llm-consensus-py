"""Thin async clients for the three reviewer LLMs.

Each `review()` returns an `AgentReview`. The model output is expected to be
a JSON object `{"score": float in [0,1], "feedback": str}`. We tolerate sloppy
formatting (extra prose, code fences) by extracting the first JSON object.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from llm_consensus.state import AgentReview

REVIEW_SYSTEM_PROMPT = """\
You are a strict code/plan reviewer in a multi-LLM consensus loop.

Reply with a SINGLE JSON object (no prose, no code fences) of this shape:
{"score": 0.0-1.0, "feedback": "<concise critique, <=120 words>"}

Score rubric:
- 0.95+ : ready to ship, nothing meaningful to add
- 0.85-0.94 : good, minor issues
- 0.70-0.84 : workable, several real issues
- <0.70 : significant problems
"""


class Reviewer(Protocol):
    name: str

    async def review(self, task: str, phase: str, output: str, round_: int) -> AgentReview: ...


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _parse(name: str, raw: str, round_: int, latency_ms: int) -> AgentReview:
    match = _JSON_RE.search(raw)
    if not match:
        return AgentReview(
            agent=name, score=0.0, feedback=f"parse_error: {raw[:200]}",
            round=round_, latency_ms=latency_ms, error="no_json",
        )
    try:
        data = json.loads(match.group(0))
        return AgentReview(
            agent=name,
            score=float(data["score"]),
            feedback=str(data.get("feedback", "")),
            round=round_,
            latency_ms=latency_ms,
        )
    except (ValueError, KeyError) as exc:
        return AgentReview(
            agent=name, score=0.0, feedback=raw[:200],
            round=round_, latency_ms=latency_ms, error=str(exc),
        )


def _build_prompt(task: str, phase: str, output: str) -> str:
    return (
        f"TASK: {task}\n\n"
        f"PHASE: {phase}\n\n"
        f"OUTPUT TO REVIEW:\n{output}\n\n"
        "Reply with the JSON object only."
    )


@dataclass
class AnthropicReviewer:
    name: str = "claude"
    model: str = "claude-sonnet-4-5"
    api_key: str | None = None

    async def review(self, task: str, phase: str, output: str, round_: int) -> AgentReview:
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 800,
                    "system": REVIEW_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": _build_prompt(task, phase, output)}],
                },
            )
        latency = int((time.perf_counter() - t0) * 1000)
        if r.status_code >= 400:
            return AgentReview(agent=self.name, score=0.0, feedback=r.text[:200],
                               round=round_, latency_ms=latency, error=f"http_{r.status_code}")
        text = "".join(block.get("text", "") for block in r.json().get("content", []))
        return _parse(self.name, text, round_, latency)


@dataclass
class OpenAICompatibleReviewer:
    """Works for OpenAI and xAI/Grok (same chat-completions schema)."""

    name: str
    model: str
    base_url: str
    env_key: str

    async def review(self, task: str, phase: str, output: str, round_: int) -> AgentReview:
        key = os.environ.get(self.env_key, "")
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                        {"role": "user", "content": _build_prompt(task, phase, output)},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 800,
                },
            )
        latency = int((time.perf_counter() - t0) * 1000)
        if r.status_code >= 400:
            return AgentReview(agent=self.name, score=0.0, feedback=r.text[:200],
                               round=round_, latency_ms=latency, error=f"http_{r.status_code}")
        text = r.json()["choices"][0]["message"]["content"]
        return _parse(self.name, text, round_, latency)


def default_reviewers() -> list[Reviewer]:
    """Return reviewers for whichever API keys are available.

    Always includes the UserPreferenceReviewer ("vote like the user"): with no
    history it abstains gracefully, but once feedback accumulates it joins
    every round from the start.
    """
    from llm_consensus.user_reviewer import UserPreferenceReviewer

    out: list[Reviewer] = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        out.append(AnthropicReviewer())
    if os.environ.get("OPENAI_API_KEY"):
        out.append(OpenAICompatibleReviewer(
            name="gpt", model="gpt-4o-mini",
            base_url="https://api.openai.com/v1", env_key="OPENAI_API_KEY",
        ))
    if os.environ.get("XAI_API_KEY"):
        out.append(OpenAICompatibleReviewer(
            name="grok", model="grok-2-latest",
            base_url="https://api.x.ai/v1", env_key="XAI_API_KEY",
        ))
    out.append(UserPreferenceReviewer())
    return out
