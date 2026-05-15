"""Shared state that flows through the LangGraph.

In LangGraph, every node receives the current state, returns a partial update,
and the framework merges those updates. For list-typed fields we use
`Annotated[..., operator.add]` so updates *append* rather than *replace*.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

Phase = Literal["plan", "code", "review"]


class AgentReview(BaseModel):
    """One reviewer's verdict for one round."""

    agent: str
    score: float = Field(ge=0.0, le=1.0)
    feedback: str
    round: int
    latency_ms: int = 0
    error: str | None = None


class RoundRecord(BaseModel):
    """A complete round of consensus: the prompt, every review, the aggregated score."""

    round: int
    output_reviewed: str
    reviews: list[AgentReview]
    aggregated_score: float
    weights: dict[str, float]


class BanditStats(BaseModel):
    """Per-agent statistics maintained by the UCB1 bandit."""

    pulls: int = 0
    reward_sum: float = 0.0

    @property
    def mean(self) -> float:
        return self.reward_sum / self.pulls if self.pulls else 0.0


class ConsensusState(TypedDict, total=False):
    """The dict that flows through every node.

    TypedDict (not pydantic) because LangGraph's state-merging semantics work
    best with plain dicts + `Annotated` reducers.
    """

    # Inputs (set once by the caller)
    session_id: str
    task: str
    phase: Phase
    current_output: str
    threshold: float
    max_rounds: int

    # Loop state (mutated each round)
    round: int
    latest_reviews: list[AgentReview]
    latest_score: float
    latest_weights: dict[str, float]

    # Accumulated over the run — `operator.add` makes appends merge correctly
    history: Annotated[list[RoundRecord], operator.add]

    # Bandit state — replaced wholesale by the aggregate node each round
    bandit: dict[str, BanditStats]

    # Debate sub-loop state
    debate_round: int
    max_debate_rounds: int
    disagreement_threshold: float

    # Terminal flags
    converged: bool
    stop_reason: str
