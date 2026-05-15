"""On-demand demo of the consensus loop using scripted reviewers.

Why this exists:
  The unit tests (`tests/`) verify each node and the wired graph, but they're
  silent — `pytest .` just prints "17 passed". This module lets you *watch*
  the system run: a deliberately-bad function, two reviewers tuned to disagree
  on round 1 (triggering the debate sub-loop), then convergence by round ~4.
  No API keys required — every reviewer is a `ScriptedReviewer`.

Run it:
  uv run llm-consensus-demo                # CLI, streams events to stdout
  uv run llm-consensus-demo --dashboard    # Also serves the live UI on :3001
  uv run llm-consensus-demo --dashboard --port 4001

The dashboard runs on **port 3001 by default** so it doesn't collide with the
main server on :3000. Open http://localhost:3001/dashboard, click
"Run Consensus", and watch the rounds render live via SSE.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field

from llm_consensus.graph import build_graph
from llm_consensus.state import AgentReview, ConsensusState

CRAP_CODE = '''\
def divide(a, b):
    """Divide two numbers."""
    return a / b
'''


@dataclass
class ScriptedReviewer:
    """Returns pre-recorded (score, feedback) pairs in order.

    Repeats the last entry if asked more times than scripted — so the demo is
    safe to re-run inside the dashboard without exploding.
    """

    name: str
    script: list[tuple[float, str]]
    calls: list[tuple[int, float]] = field(default_factory=list)

    async def review(self, task: str, phase: str, output: str, round_: int) -> AgentReview:
        idx = min(len(self.calls), len(self.script) - 1)
        score, feedback = self.script[idx]
        self.calls.append((round_, score))
        return AgentReview(
            agent=self.name, score=score, feedback=feedback,
            round=round_, latency_ms=0,
        )


def make_scripted_reviewers() -> list[ScriptedReviewer]:
    """Two reviewers tuned to disagree → debate → converge.

    Script entries are consumed in order across dispatch *and* debate calls.
    The progression below produces:
      Round 1 dispatch: grok 0.85 vs gpt 0.40   → gap 0.45 → DEBATE
      Round 1 debate1:  grok 0.72 vs gpt 0.55   → gap 0.17 → DEBATE
      Round 1 debate2:  grok 0.68 vs gpt 0.62   → gap 0.06 → AGGREGATE (~0.65)
      Round 2 dispatch: grok 0.88 vs gpt 0.78   → gap 0.10 → AGGREGATE (~0.83)
      Round 3 dispatch: grok 0.96 vs gpt 0.92   → no debate → AGGREGATE (~0.94)
      Round 4 dispatch: grok 0.97 vs gpt 0.96   → no debate → CONVERGED (~0.965)
    """
    grok = ScriptedReviewer(
        name="grok",
        script=[
            (0.85, "Looks fine — division is straightforward."),
            (0.72, "Fair point on zero-division. Conceding partially."),
            (0.68, "Type hints would help, but core logic is OK."),
            (0.88, "Guard added — good. Docstring still thin."),
            (0.96, "Tight. Maybe add an example to the docstring."),
            (0.97, "Ship it."),
        ],
    )
    gpt = ScriptedReviewer(
        name="gpt",
        script=[
            (0.40, "No input validation, no zero-division guard. Will crash."),
            (0.55, "Holding firm — division by zero is a real bug."),
            (0.62, "Acceptable compromise. Types still missing."),
            (0.78, "Type hints added, division guarded. Better."),
            (0.92, "Minor: docstring lacks examples."),
            (0.96, "Looks good."),
        ],
    )
    return [grok, gpt]


def _initial_state() -> ConsensusState:
    return {
        "session_id": "demo",
        "task": "Review this Python `divide` function for production-readiness.",
        "phase": "code",
        "current_output": CRAP_CODE,
        "threshold": 0.95,
        "max_rounds": 10,
        "max_debate_rounds": 2,
        "disagreement_threshold": 0.10,
        "round": 0,
        "debate_round": 0,
        "history": [],
        "converged": False,
    }


def _color(s: str, code: str) -> str:
    return f"\033[{code}m{s}\033[0m"


def _print_event(node: str, update: dict) -> None:
    if node == "reset_debate":
        return
    if node == "dispatch":
        round_ = update.get("round")
        print(f"\n{_color(f'── Round {round_} ── dispatch', '1;36')}")
        for r in update.get("latest_reviews", []) or []:
            color = "32" if r.score >= 0.85 else "33" if r.score >= 0.6 else "31"
            print(f"  {r.agent:>6}  {_color(f'{r.score:.2f}', color)}  {r.feedback}")
    elif node == "debate":
        dr = update.get("debate_round")
        print(f"\n  {_color(f'  debate round {dr} — exchanging critiques', '35')}")
        for r in update.get("latest_reviews", []) or []:
            color = "32" if r.score >= 0.85 else "33" if r.score >= 0.6 else "31"
            print(f"    {r.agent:>6}  {_color(f'{r.score:.2f}', color)}  {r.feedback}")
    elif node == "aggregate":
        score = update.get("latest_score", 0.0)
        weights = update.get("latest_weights", {}) or {}
        wstr = "  ".join(f"{k}={v:.2f}" for k, v in weights.items())
        color = "32" if score >= 0.95 else "33" if score >= 0.7 else "31"
        print(f"  aggregate: weighted score = {_color(f'{score:.3f}', color)}   "
              f"weights: {wstr}")
    elif node == "refine":
        print(f"  refine: critiques appended → next round")
    elif node == "converged":
        print(f"\n{_color('CONVERGED', '1;32')} — {update.get('stop_reason')}")
    elif node == "exhausted":
        print(f"\n{_color('EXHAUSTED', '1;31')} — {update.get('stop_reason')}")


async def run_cli() -> None:
    reviewers = make_scripted_reviewers()
    graph = build_graph(reviewers=reviewers)

    print(_color("LLM CONSENSUS — DEMO (scripted reviewers, no API keys used)", "1"))
    print(f"Reviewers : {', '.join(r.name for r in reviewers)}")
    print("Threshold : 0.95   Max rounds : 10   Disagreement threshold : 0.10")
    print(f"Output to review:\n{CRAP_CODE}")

    async for chunk in graph.astream(
        _initial_state(), {"recursion_limit": 50}, stream_mode="updates",
    ):
        for node, update in chunk.items():
            _print_event(node, update)

    print()


def run_with_dashboard(port: int) -> None:
    """Start the FastAPI server on `port` with a graph pre-built from scripted reviewers.

    Monkey-patches `server._graph` before uvicorn boots so any incoming request
    to /consensus or /consensus/stream uses the scripted reviewers — same code
    path as production, just deterministic inputs.
    """
    import uvicorn

    from llm_consensus import server as srv

    srv._graph = build_graph(reviewers=make_scripted_reviewers())

    print(_color(f"Demo dashboard → http://localhost:{port}/dashboard", "1;36"))
    print("Click 'Run Consensus' in the UI to fire a scripted demo round.")
    print("(Reviewers are scripted; no API keys required.)")
    uvicorn.run(srv.app, host="127.0.0.1", port=port, reload=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-consensus-demo",
        description="Watch the consensus loop run with scripted reviewers — no API keys needed.",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Also serve the live SSE dashboard (default port 3001).",
    )
    parser.add_argument(
        "--port", type=int, default=3001,
        help="Port for --dashboard mode (default: 3001, kept distinct from main server's 3000).",
    )
    args = parser.parse_args()

    if args.dashboard:
        run_with_dashboard(args.port)
    else:
        asyncio.run(run_cli())


if __name__ == "__main__":
    main()
