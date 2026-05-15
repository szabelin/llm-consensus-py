"""Decide node: conditional edge function — choose `refine` or END based on state.

Conditional edges are how LangGraph expresses branching. The function returns
a STRING that names the next node (or `END`). LangGraph looks it up in the
mapping passed to `add_conditional_edges`.
"""

from __future__ import annotations

from langgraph.graph import END

from llm_consensus.state import ConsensusState


def decide_route(state: ConsensusState) -> str:
    score = state.get("latest_score", 0.0)
    threshold = state.get("threshold", 0.95)
    round_ = state.get("round", 0)
    max_rounds = state.get("max_rounds", 10)

    if score >= threshold:
        return "converged"
    if round_ >= max_rounds:
        return "exhausted"
    return "refine"


def finalise_converged(state: ConsensusState) -> dict:
    return {"converged": True, "stop_reason": "threshold_met"}


def finalise_exhausted(state: ConsensusState) -> dict:
    return {"converged": False, "stop_reason": "max_rounds"}


__all__ = ["decide_route", "finalise_converged", "finalise_exhausted", "END"]
