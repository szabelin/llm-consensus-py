"""LLM Consensus — a LangGraph orchestration of multiple LLMs reaching agreement."""

from llm_consensus.graph import build_graph
from llm_consensus.state import ConsensusState

__all__ = ["build_graph", "ConsensusState"]


def main() -> None:
    from llm_consensus.server import run

    run()
