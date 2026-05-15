"""Refine node: synthesise reviewer feedback into a new prompt for next round.

In a full system you'd call an LLM here to rewrite `current_output` based on
the critiques. For the orchestration MVP we just *augment* the prompt with the
top-weighted critiques, leaving the actual rewrite to the upstream caller
(typically Claude Code, which originally produced the output).
"""

from __future__ import annotations

from llm_consensus.state import ConsensusState


async def refine_node(state: ConsensusState) -> dict:
    reviews = state["latest_reviews"]
    weights = state.get("latest_weights") or {}

    ordered = sorted(reviews, key=lambda r: weights.get(r.agent, 0.0), reverse=True)
    critique_block = "\n".join(
        f"- [{r.agent} score={r.score:.2f} weight={weights.get(r.agent, 0):.2f}] {r.feedback}"
        for r in ordered
    )

    revised = (
        f"{state['current_output']}\n\n"
        f"--- ROUND {state['round']} CRITIQUES ---\n{critique_block}\n"
        f"--- END CRITIQUES ---\n"
    )

    return {"current_output": revised}
