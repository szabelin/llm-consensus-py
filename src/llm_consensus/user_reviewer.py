"""UserPreferenceReviewer — the 4th agent that votes the way YOU vote.

How it works:
1. Reads recent feedback records from the PreferenceStore.
2. Builds a few-shot prompt: "here are outputs the user has commented on, and
   what they said about each. Given a NEW output, predict the user's score
   (0-1) and what they'd say about it."
3. Calls Claude to produce that prediction. The prediction is returned in the
   same `AgentReview` shape as Claude/GPT/Grok, so the graph treats it
   identically.

Bootstrap behaviour: with zero feedback records, it abstains (returns the
neutral score 0.5 with no exploration weight) so it doesn't bias the room
before it has anything to learn from.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import httpx

from llm_consensus.preferences import PreferenceStore
from llm_consensus.state import AgentReview

_PREDICT_SYSTEM_PROMPT = """\
You are a model of a specific user's coding preferences. You have read their
historical feedback on past outputs (provided below). When shown a NEW output,
predict the score (0-1) and one-line critique THIS USER would give.

Return a SINGLE JSON object: {"score": 0.0-1.0, "feedback": "<<=120 words>"}.
Channel the user's style, biases, and pet peeves. Be specific.
"""

_JSON_RE = re.compile(r"\{[\s\S]*\}")


@dataclass
class UserPreferenceReviewer:
    name: str = "user_pref"
    model: str = "claude-sonnet-4-5"
    store: PreferenceStore | None = None
    api_key: str | None = None

    def __post_init__(self) -> None:
        if self.store is None:
            self.store = PreferenceStore()

    def _few_shot_block(self) -> str:
        records = self.store.recent(20)
        if not records:
            return ""
        examples = "\n\n".join(
            f"PAST OUTPUT:\n{r.output_snippet[:400]}\n"
            f"USER SAID: {r.comment}"
            for r in records
        )
        return f"--- USER'S HISTORICAL FEEDBACK ---\n{examples}\n--- END HISTORY ---\n"

    async def review(self, task: str, phase: str, output: str, round_: int) -> AgentReview:
        # Cold start: no history → abstain with a neutral score.
        history = self._few_shot_block()
        if not history:
            return AgentReview(
                agent=self.name, score=0.5, round=round_,
                feedback="(no user feedback yet — cold start, abstaining)",
                latency_ms=0,
            )

        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            return AgentReview(
                agent=self.name, score=0.5, round=round_,
                feedback="(ANTHROPIC_API_KEY missing — preference agent disabled)",
                error="no_key", latency_ms=0,
            )

        prompt = (
            f"{history}\n"
            f"TASK: {task}\nPHASE: {phase}\n\nNEW OUTPUT:\n{output}\n\n"
            "Predict the user's JSON verdict."
        )

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
                    "max_tokens": 600,
                    "system": _PREDICT_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
        latency = int((time.perf_counter() - t0) * 1000)
        if r.status_code >= 400:
            return AgentReview(agent=self.name, score=0.5, feedback=r.text[:200],
                               round=round_, latency_ms=latency, error=f"http_{r.status_code}")
        text = "".join(b.get("text", "") for b in r.json().get("content", []))
        m = _JSON_RE.search(text)
        if not m:
            return AgentReview(agent=self.name, score=0.5, feedback=text[:200],
                               round=round_, latency_ms=latency, error="no_json")
        try:
            data = json.loads(m.group(0))
            return AgentReview(
                agent=self.name,
                score=float(data["score"]),
                feedback=str(data.get("feedback", "")),
                round=round_, latency_ms=latency,
            )
        except (ValueError, KeyError) as e:
            return AgentReview(agent=self.name, score=0.5, feedback=text[:200],
                               round=round_, latency_ms=latency, error=str(e))
