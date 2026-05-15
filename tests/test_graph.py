"""End-to-end test of the compiled LangGraph with mocked reviewers.

Covers:
- Happy path: reviewers agree → converges in one round.
- Debate sub-loop: reviewers disagree → debate fires → next round agrees.
- Exhaustion: reviewers never converge → stops at max_rounds.
"""

import pytest

from llm_consensus.graph import build_graph

from tests.conftest import FakeReviewer


def _state(**overrides):
    base = {
        "task": "t",
        "phase": "review",
        "current_output": "code",
        "threshold": 0.95,
        "max_rounds": 5,
        "max_debate_rounds": 2,
        "disagreement_threshold": 0.10,
        "round": 0,
        "debate_round": 0,
        "history": [],
        "converged": False,
        "session_id": "test",
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_converges_when_reviewers_agree():
    reviewers = [FakeReviewer("a", [0.97]), FakeReviewer("b", [0.96])]
    graph = build_graph(reviewers)
    final = await graph.ainvoke(_state(), {"recursion_limit": 50})
    assert final["converged"] is True
    assert final["stop_reason"] == "threshold_met"
    assert final["round"] == 1


@pytest.mark.asyncio
async def test_debate_fires_on_disagreement_then_converges():
    # Round 1: huge gap (0.95 vs 0.50). After debate they revise to (0.95, 0.94).
    a = FakeReviewer("a", [0.95, 0.96])
    b = FakeReviewer("b", [0.50, 0.96])  # second call = after debate prompt
    graph = build_graph([a, b])
    final = await graph.ainvoke(_state(), {"recursion_limit": 50})
    # b's second call is the debate revision, which should have been triggered.
    assert len(b.calls) >= 2
    assert final["converged"] is True


@pytest.mark.asyncio
async def test_exhausts_when_no_convergence():
    reviewers = [FakeReviewer("a", [0.6]), FakeReviewer("b", [0.6])]
    graph = build_graph(reviewers)
    final = await graph.ainvoke(_state(max_rounds=3), {"recursion_limit": 50})
    assert final["converged"] is False
    assert final["stop_reason"] == "max_rounds"
    assert final["round"] == 3


@pytest.mark.asyncio
async def test_history_records_every_round():
    a = FakeReviewer("a", [0.7, 0.8, 0.97])
    b = FakeReviewer("b", [0.75, 0.85, 0.97])
    graph = build_graph([a, b])
    final = await graph.ainvoke(_state(), {"recursion_limit": 50})
    assert len(final["history"]) == final["round"]
    # Scores should be monotonically rising thanks to scripted feedback.
    scores = [h.aggregated_score for h in final["history"]]
    assert scores == sorted(scores)
