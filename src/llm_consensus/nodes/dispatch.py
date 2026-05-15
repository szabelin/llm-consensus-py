"""Dispatch node: fan out the current output to every reviewer in parallel."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from llm_consensus.agents import Reviewer
from llm_consensus.policy import bandit
from llm_consensus.state import ConsensusState

DispatchNode = Callable[[ConsensusState], Awaitable[dict]]


def make_dispatch_node(reviewers: list[Reviewer]) -> DispatchNode:
    """Build a node closed over the reviewer set.

    Why a factory? Nodes are plain functions; LangGraph doesn't pass services
    in. Closing over `reviewers` keeps the node signature simple while still
    being testable (just call `make_dispatch_node([fake_reviewer])`).
    """

    async def dispatch(state: ConsensusState) -> dict:
        round_ = state.get("round", 0) + 1
        bandit_state = state.get("bandit") or bandit.initialise([r.name for r in reviewers])

        coros = [
            r.review(state["task"], state["phase"], state["current_output"], round_)
            for r in reviewers
        ]
        results = await asyncio.gather(*coros)

        return {
            "round": round_,
            "latest_reviews": list(results),
            "bandit": bandit_state,
        }

    return dispatch
