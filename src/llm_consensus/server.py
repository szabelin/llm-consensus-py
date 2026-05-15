"""FastAPI server.

Endpoints:
- POST /consensus    — start a consensus run, returns the final state.
- POST /consensus/stream — same, but streams every node update as SSE.
- POST /feedback     — record free-text user feedback against a session/round.
- GET  /dashboard    — minimal HTML monitor.
- GET  /health       — health check.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from llm_consensus.graph import build_graph
from llm_consensus.preferences import FeedbackRecord, PreferenceStore
from llm_consensus.state import ConsensusState

load_dotenv()

app = FastAPI(title="LLM Consensus", version="0.1.0")
_preferences = PreferenceStore()
_graph = None  # lazy-built — needs env vars at request time


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


class ConsensusRequest(BaseModel):
    task: str
    phase: str = "review"
    current_output: str
    threshold: float = 0.95
    max_rounds: int = 10
    max_debate_rounds: int = 2
    disagreement_threshold: float = 0.10
    session_id: str = Field(default_factory=lambda: str(uuid4()))


class FeedbackRequest(BaseModel):
    session_id: str
    round: int
    output_snippet: str
    comment: str


def _initial_state(req: ConsensusRequest) -> ConsensusState:
    return {
        "session_id": req.session_id,
        "task": req.task,
        "phase": req.phase,  # type: ignore[typeddict-item]
        "current_output": req.current_output,
        "threshold": req.threshold,
        "max_rounds": req.max_rounds,
        "max_debate_rounds": req.max_debate_rounds,
        "disagreement_threshold": req.disagreement_threshold,
        "round": 0,
        "debate_round": 0,
        "history": [],
        "converged": False,
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "feedback_records": len(_preferences.all())}


@app.post("/consensus")
async def consensus(req: ConsensusRequest):
    try:
        graph = _get_graph()
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    final = await graph.ainvoke(_initial_state(req), {"recursion_limit": 50})
    return JSONResponse(_serialise(final))


@app.post("/consensus/stream")
async def consensus_stream(req: ConsensusRequest):
    try:
        graph = _get_graph()
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    async def events() -> AsyncIterator[dict]:
        async for chunk in graph.astream(
            _initial_state(req),
            {"recursion_limit": 50},
            stream_mode="updates",
        ):
            for node, update in chunk.items():
                yield {
                    "event": "node",
                    "data": json.dumps({"node": node, "update": _serialise(update)}),
                }
            await asyncio.sleep(0)
        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(events())


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    record = FeedbackRecord(
        session_id=req.session_id,
        round=req.round,
        output_snippet=req.output_snippet,
        comment=req.comment,
    )
    _preferences.append(record)
    return {"ok": True, "total": len(_preferences.all())}


@app.get("/dashboard")
async def dashboard():
    path = Path(__file__).parent / "dashboard" / "index.html"
    return FileResponse(path)


def _serialise(obj):
    """Pydantic models → dicts so JSON encoding works."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    return obj


def run() -> None:
    import uvicorn

    uvicorn.run(
        "llm_consensus.server:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "3000")),
        reload=False,
    )


if __name__ == "__main__":
    run()
