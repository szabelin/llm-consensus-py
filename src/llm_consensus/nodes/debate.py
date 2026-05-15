"""Debate sub-loop.

When two LLM reviewers disagree sharply, give each the other's critique and
ask them to revise. This is a *cycle within a cycle* — a classic LangGraph
pattern: the debate node either loops back to itself or escapes to aggregate.

The user-preference agent (`user_pref`) does not debate — its score reflects
the human, not a negotiable opinion.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from llm_consensus.agents import Reviewer
from llm_consensus.state import AgentReview, ConsensusState

DebateNode = Callable[[ConsensusState], Awaitable[dict]]
DEBATE_AGENT_PREFIX = "user_pref"


def _llm_reviews(reviews: list[AgentReview]) -> list[AgentReview]:
    return [r for r in reviews if not r.agent.startswith(DEBATE_AGENT_PREFIX)]


def max_gap(reviews: list[AgentReview]) -> tuple[float, AgentReview | None, AgentReview | None]:
    """Return (gap, hi, lo) for the most-disagreeing pair of LLM reviewers."""
    pool = _llm_reviews(reviews)
    if len(pool) < 2:
        return 0.0, None, None
    hi = max(pool, key=lambda r: r.score)
    lo = min(pool, key=lambda r: r.score)
    return hi.score - lo.score, hi, lo


def should_debate(state: ConsensusState) -> str:
    """Conditional edge after dispatch.

    Returns 'debate' if the worst-disagreeing pair exceeds the threshold AND
    we haven't burned the debate budget, else 'aggregate'.
    """
    gap, _, _ = max_gap(state["latest_reviews"])
    threshold = state.get("disagreement_threshold", 0.10)
    used = state.get("debate_round", 0)
    budget = state.get("max_debate_rounds", 2)
    if gap > threshold and used < budget:
        return "debate"
    return "aggregate"


def make_debate_node(reviewers: list[Reviewer]) -> DebateNode:
    """Build the debate node closed over the reviewer set."""
    by_name = {r.name: r for r in reviewers}

    async def debate(state: ConsensusState) -> dict:
        reviews = state["latest_reviews"]
        gap, hi, lo = max_gap(reviews)
        if hi is None or lo is None:
            return {"debate_round": state.get("debate_round", 0) + 1}

        # Build re-prompts: each side sees the other's argument and is asked
        # to either revise or stand firm with stronger reasoning.
        revised_output_for_hi = (
            f"{state['current_output']}\n\n"
            f"--- OPPOSING VIEW from {lo.agent} (score={lo.score:.2f}) ---\n"
            f"{lo.feedback}\n--- END ---\n"
            f"You previously scored this {hi.score:.2f}. Reconsider. "
            "Either revise your score (justify what you missed) or stand firm "
            "(refute the opposing point). Reply with the JSON object only."
        )
        revised_output_for_lo = (
            f"{state['current_output']}\n\n"
            f"--- OPPOSING VIEW from {hi.agent} (score={hi.score:.2f}) ---\n"
            f"{hi.feedback}\n--- END ---\n"
            f"You previously scored this {lo.score:.2f}. Reconsider. "
            "Either revise your score (justify what you missed) or stand firm "
            "(refute the opposing point). Reply with the JSON object only."
        )

        debate_round = state.get("debate_round", 0) + 1

        hi_agent = by_name.get(hi.agent)
        lo_agent = by_name.get(lo.agent)
        # If a reviewer can't be found (unlikely), keep their prior review.
        coros = []
        if hi_agent is not None:
            coros.append(hi_agent.review(
                state["task"], state["phase"], revised_output_for_hi, state["round"],
            ))
        if lo_agent is not None:
            coros.append(lo_agent.review(
                state["task"], state["phase"], revised_output_for_lo, state["round"],
            ))
        revised = await asyncio.gather(*coros) if coros else []

        # Replace the prior reviews from hi/lo with the new ones; everyone
        # else carries over unchanged.
        revised_by_name = {r.agent: r for r in revised}
        merged = [revised_by_name.get(r.agent, r) for r in reviews]

        return {
            "latest_reviews": merged,
            "debate_round": debate_round,
        }

    return debate
