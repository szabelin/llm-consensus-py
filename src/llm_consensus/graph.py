"""LangGraph wiring.

Topology (outer loop = consensus rounds; inner loop = debate sub-rounds):

        ┌──────────┐
        │ dispatch │  fan out to every reviewer in parallel (incl. user_pref)
        └────┬─────┘
             ▼
       ┌─────────────┐
       │ should_     │  CONDITIONAL: gap > threshold AND budget remains?
       │ debate?     │      yes → debate     no → aggregate
       └──┬───────┬──┘
          │       │
          ▼       ▼
     ┌────────┐  ┌──────────┐
     │ debate │  │aggregate │  weighted-average via UCB1 bandit weights
     └──┬─────┘  └────┬─────┘
        │             │
        └──► (loop back to should_debate)
                      ▼
                ┌──────────┐
                │  decide  │  CONDITIONAL: converged / exhausted / refine
                └──┬───┬─┬─┘
       converged  │   │  │  exhausted
                  ▼   ▼  ▼
              [END]    refine
                        │
                        └──► (loop back to dispatch — outer iteration)
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from llm_consensus.agents import Reviewer, default_reviewers
from llm_consensus.nodes import (
    aggregate_node,
    decide_route,
    make_debate_node,
    make_dispatch_node,
    refine_node,
    should_debate,
)
from llm_consensus.nodes.decide import finalise_converged, finalise_exhausted
from llm_consensus.state import ConsensusState


async def _reset_debate(state: ConsensusState) -> dict:
    """Tiny passthrough that resets the debate counter at the start of each round.

    Without this, the debate budget would persist across the outer-loop
    refine→dispatch cycle and prevent debate in later rounds.
    """
    return {"debate_round": 0}


def build_graph(reviewers: list[Reviewer] | None = None):
    """Construct + compile the LangGraph.

    Returns a compiled graph exposing `.ainvoke(state)` and `.astream(state)`.
    """
    reviewers = reviewers if reviewers is not None else default_reviewers()
    if not reviewers:
        raise RuntimeError(
            "No reviewers available — set ANTHROPIC_API_KEY, OPENAI_API_KEY, or XAI_API_KEY."
        )

    graph = StateGraph(ConsensusState)

    graph.add_node("reset_debate", _reset_debate)
    graph.add_node("dispatch", make_dispatch_node(reviewers))
    graph.add_node("debate", make_debate_node(reviewers))
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("refine", refine_node)
    graph.add_node("converged", finalise_converged)
    graph.add_node("exhausted", finalise_exhausted)

    graph.set_entry_point("reset_debate")
    graph.add_edge("reset_debate", "dispatch")

    graph.add_conditional_edges(
        "dispatch",
        should_debate,
        {"debate": "debate", "aggregate": "aggregate"},
    )
    # After debate, re-check disagreement — may loop back to debate or escape.
    graph.add_conditional_edges(
        "debate",
        should_debate,
        {"debate": "debate", "aggregate": "aggregate"},
    )

    graph.add_conditional_edges(
        "aggregate",
        decide_route,
        {"converged": "converged", "exhausted": "exhausted", "refine": "refine"},
    )
    graph.add_edge("refine", "reset_debate")
    graph.add_edge("converged", END)
    graph.add_edge("exhausted", END)

    # `recursion_limit` defaults to 25 — generous for our 10 outer * 3 inner.
    return graph.compile()
