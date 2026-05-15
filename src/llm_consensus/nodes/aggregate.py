"""Aggregate node: combine reviewer scores into a single number using bandit weights."""

from __future__ import annotations

from llm_consensus.policy import bandit as bandit_policy
from llm_consensus.state import ConsensusState, RoundRecord


async def aggregate_node(state: ConsensusState) -> dict:
    reviews = state["latest_reviews"]
    bandit_state = state["bandit"]

    # Compute weights *before* recording this round's reward; otherwise the
    # current observation would influence its own weighting.
    weights = bandit_policy.weights(bandit_state)

    total_weight = sum(weights.get(r.agent, 0.0) for r in reviews) or 1.0
    aggregated = sum(r.score * weights.get(r.agent, 0.0) for r in reviews) / total_weight

    new_bandit = bandit_policy.update(
        bandit_state,
        scores={r.agent: r.score for r in reviews},
        aggregated=aggregated,
    )

    record = RoundRecord(
        round=state["round"],
        output_reviewed=state["current_output"],
        reviews=reviews,
        aggregated_score=aggregated,
        weights=weights,
    )

    return {
        "latest_score": aggregated,
        "latest_weights": weights,
        "bandit": new_bandit,
        "history": [record],  # `Annotated[..., operator.add]` makes this append
    }
