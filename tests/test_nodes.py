import pytest

from llm_consensus.nodes.aggregate import aggregate_node
from llm_consensus.nodes.debate import max_gap, should_debate
from llm_consensus.nodes.decide import decide_route
from llm_consensus.nodes.dispatch import make_dispatch_node
from llm_consensus.policy import bandit
from llm_consensus.state import AgentReview

from tests.conftest import FakeReviewer


@pytest.mark.asyncio
async def test_dispatch_fans_out_to_all_reviewers():
    reviewers = [FakeReviewer("a", [0.8]), FakeReviewer("b", [0.6])]
    node = make_dispatch_node(reviewers)
    state = {
        "task": "t", "phase": "review", "current_output": "code",
        "round": 0,
    }
    update = await node(state)
    assert update["round"] == 1
    assert {r.agent for r in update["latest_reviews"]} == {"a", "b"}
    assert set(update["bandit"].keys()) == {"a", "b"}


@pytest.mark.asyncio
async def test_aggregate_uses_bandit_weights_and_appends_history():
    state = {
        "round": 1,
        "current_output": "x",
        "latest_reviews": [
            AgentReview(agent="a", score=0.9, feedback="", round=1),
            AgentReview(agent="b", score=0.5, feedback="", round=1),
        ],
        "bandit": bandit.initialise(["a", "b"]),  # uniform 0.5/0.5
    }
    update = await aggregate_node(state)
    assert abs(update["latest_score"] - 0.7) < 1e-9
    assert len(update["history"]) == 1


def test_should_debate_triggers_on_large_gap():
    state = {
        "latest_reviews": [
            AgentReview(agent="a", score=0.9, feedback="", round=1),
            AgentReview(agent="b", score=0.5, feedback="", round=1),
        ],
        "disagreement_threshold": 0.10,
        "debate_round": 0,
        "max_debate_rounds": 2,
    }
    assert should_debate(state) == "debate"


def test_should_debate_skipped_when_budget_exhausted():
    state = {
        "latest_reviews": [
            AgentReview(agent="a", score=0.9, feedback="", round=1),
            AgentReview(agent="b", score=0.5, feedback="", round=1),
        ],
        "disagreement_threshold": 0.10,
        "debate_round": 2,
        "max_debate_rounds": 2,
    }
    assert should_debate(state) == "aggregate"


def test_should_debate_ignores_user_pref_agent():
    state = {
        "latest_reviews": [
            AgentReview(agent="a", score=0.9, feedback="", round=1),
            AgentReview(agent="b", score=0.85, feedback="", round=1),
            AgentReview(agent="user_pref", score=0.2, feedback="", round=1),
        ],
        "disagreement_threshold": 0.10,
        "debate_round": 0,
        "max_debate_rounds": 2,
    }
    gap, hi, lo = max_gap(state["latest_reviews"])
    assert hi.agent != "user_pref" and lo.agent != "user_pref"
    assert should_debate(state) == "aggregate"  # only LLM gap is 0.05


def test_decide_routes_converged_at_threshold():
    state = {"latest_score": 0.96, "threshold": 0.95, "round": 2, "max_rounds": 10}
    assert decide_route(state) == "converged"


def test_decide_routes_exhausted_at_max_rounds():
    state = {"latest_score": 0.5, "threshold": 0.95, "round": 10, "max_rounds": 10}
    assert decide_route(state) == "exhausted"


def test_decide_routes_refine_otherwise():
    state = {"latest_score": 0.8, "threshold": 0.95, "round": 3, "max_rounds": 10}
    assert decide_route(state) == "refine"
