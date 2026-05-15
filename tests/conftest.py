"""Fake reviewer that returns scripted scores — lets us test the graph offline."""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_consensus.state import AgentReview


@dataclass
class FakeReviewer:
    name: str
    scores: list[float]
    feedback: str = "fake"
    calls: list[tuple[str, int]] = field(default_factory=list)

    async def review(self, task: str, phase: str, output: str, round_: int) -> AgentReview:
        idx = min(len(self.calls), len(self.scores) - 1)
        self.calls.append((output[:40], round_))
        return AgentReview(
            agent=self.name, score=self.scores[idx], feedback=self.feedback,
            round=round_, latency_ms=0,
        )
