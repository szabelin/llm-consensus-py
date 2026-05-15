# Consensus Workflow Instructions

CRITICAL: These are your operational instructions. Read fully before any complex task.

## What This Is

You have access to a consensus server (LangGraph-based, Python) at
http://localhost:3000 that gets independent reviews from multiple LLMs in
parallel — typically ChatGPT and Grok — plus an optional user-preference
agent that votes the way the user historically votes. Reviewers' verdicts
are weighted by a UCB1 multi-armed bandit, and a debate sub-loop fires when
two reviewers disagree sharply. The system catches errors and edge cases
before you ship them to the user.

## Mandatory Rules

1. Call the consensus server for EVERY iteration on a non-trivial task — no exceptions
2. Display ALL consensus details to the user EVERY round (per-agent scores, feedback, aggregated score, status)
3. The aggregated score must REACH the threshold (default 0.95) to be considered converged
4. Maximum 10 rounds per task — if `stop_reason: "max_rounds"` comes back, stop and ask the user
5. Never silently skip consensus for complex tasks

## When You Must Seek Consensus

- Plans with 3+ steps
- Code over 50 lines
- Architecture or interface decisions
- Complex algorithms (concurrency, parsing, anything stateful)
- Security-sensitive code
- Anything the user explicitly requests review for

## When You May Skip Consensus

- Simple one-liner fixes
- Typo corrections
- Adding comments / renames / reformatting
- Trivial follow-ups already implied by an approved plan

## Server Endpoints

Health check:
```
curl http://localhost:3000/health
```

Consensus request (run-to-completion):
```
curl -s -X POST "http://localhost:3000/consensus" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "unique-id-per-task",
    "task": "Description of what you are building",
    "phase": "plan",
    "current_output": "Your plan or code as a string",
    "threshold": 0.95,
    "max_rounds": 10,
    "max_debate_rounds": 2,
    "disagreement_threshold": 0.10
  }'
```

Consensus request (streaming, Server-Sent Events — recommended for long runs):
```
curl -N -X POST "http://localhost:3000/consensus/stream" \
  -H "Content-Type: application/json" \
  -d '{ ...same body... }'
```

The streaming endpoint emits one `event: node` per LangGraph step
(`dispatch`, `debate`, `aggregate`, `refine`, `converged`/`exhausted`) and
finishes with `event: done`. Use it whenever you want round-by-round
visibility.

Submit user feedback to train the user-preference agent:
```
curl -s -X POST "http://localhost:3000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "task-123",
    "round": 3,
    "output_snippet": "the code or plan snippet the user is commenting on",
    "comment": "Too verbose. I prefer no docstrings on one-line helpers."
  }'
```

Live dashboard for the user: http://localhost:3000/dashboard

## Request Fields

| Field | Required | Default | Notes |
|---|---|---|---|
| `session_id` | optional | random UUID | reuse across rounds of the same phase for the same task |
| `task` | yes | — | short description of what you're building |
| `phase` | optional | "review" | one of "plan", "code", "review" |
| `current_output` | yes | — | the plan, code, or content to be reviewed |
| `threshold` | optional | 0.95 | aggregated score required to converge |
| `max_rounds` | optional | 10 | outer-loop budget |
| `max_debate_rounds` | optional | 2 | inner debate sub-loop budget per outer round |
| `disagreement_threshold` | optional | 0.10 | max-min reviewer-score gap that triggers debate |

## Response Fields (POST /consensus)

The server returns the **final `ConsensusState`** after the run, including:

- `converged` (bool): true if threshold met
- `stop_reason`: `"threshold_met"` or `"max_rounds"`
- `latest_score`: the final aggregated score
- `latest_weights`: the bandit weights used in the final round (per reviewer)
- `latest_reviews`: array of `{agent, score, feedback, round, latency_ms, error?}`
- `round`: the last round number executed
- `history`: array of `RoundRecord`s — one per round, each with the output
  reviewed, every reviewer's review, the aggregated score, and the weights

For the streaming endpoint, each `event: node` carries
`{node: <name>, update: <partial state>}` — accumulate them client-side to
reconstruct the same picture.

## Your Workflow

### Phase 1: Plan

1. Receive the task from the user.
2. Generate a structured plan (numbered steps).
3. Call `POST /consensus` (or `/consensus/stream`) with `phase: "plan"`.
4. IMMEDIATELY display the full consensus details to the user:
   - Round number
   - Each agent's name, score, feedback
   - Aggregated score + per-agent bandit weights
   - Whether the run converged or exhausted
5. If not converged:
   - List the specific changes you're making in response to the critiques
   - Revise the plan
   - Call consensus again with the SAME `session_id` — MANDATORY
   - Display results again
6. If converged: proceed to the code phase

### Phase 2: Code

1. Write the code following the approved plan.
2. Call `POST /consensus` with `phase: "code"`.
3. Display full consensus details to the user.
4. Iterate until converged or exhausted; call the server every round.

### Phase 3: Verify (optional)

1. For complex code, ask for a verification pass.
2. Call `POST /consensus` with `phase: "review"`.
3. Include any test results / diffs in `current_output`.
4. Display results to the user.

## How to Display Consensus Results

After every consensus response, show the user something like:

```
CONSENSUS ROUND {round}/{max_rounds}

ChatGPT (score: {score}, weight: {weight}): {feedback}
Grok    (score: {score}, weight: {weight}): {feedback}
[user_pref (score: {score}): {feedback}]    # only show if not abstaining

Aggregated Score: {latest_score} (threshold: {threshold})
Status: {CONVERGED or NOT YET CONVERGED — iterating}
```

If a `debate` event fired during the round, also surface that:

```
Debate fired (gap was {pre_gap:.2f}, threshold {disagreement_threshold:.2f}):
  Round 1 of debate: ChatGPT {old}→{new}, Grok {old}→{new}
  ...
Debate exited because: gap closed | budget exhausted
```

If not converged, also show what you'll change next:

```
Changes I am making based on feedback:
- {change 1 addressing specific feedback}
- {change 2 addressing specific feedback}
```

## Handling Exhaustion

When the server returns `stop_reason: "max_rounds"`:

```
CONSENSUS EXHAUSTED — {round}/{max_rounds} rounds without convergence

Final aggregated score: {latest_score} (threshold: {threshold})

Persistent issues (raised in 2+ rounds):
- {issue 1}
- {issue 2}

Options:
1. Proceed anyway (you accept the risks)
2. Take a different approach to the same task
3. Lower the threshold for this specific task
4. Abandon the task

What would you like to do?
```

Wait for the user's choice before proceeding. The Python server does not have
a `/consensus/override` endpoint — instead, *not iterating further* is the
override.

## Session ID Convention

Use descriptive IDs: `"json-parser-plan-v1"`, `"auth-system-code-v2"`.

- Reuse the same `session_id` across rounds of the same phase
- Create a new `session_id` when starting a new phase or a new task
- This matters for the user-preference agent: it ties feedback records to a
  session_id + round, and uses them as few-shot examples for future
  predictions

## Example Flow

```
User: "Build a rate limiter for our API"

[Generate plan]

[Call POST /consensus with phase=plan, session_id=rate-limiter-plan-v1]

CONSENSUS ROUND 1/10

ChatGPT (score: 0.80, weight: 0.50): "Consider distributed rate limiting with Redis"
Grok    (score: 0.84, weight: 0.50): "Add burst handling mechanism"

Aggregated Score: 0.82 (threshold: 0.95)
Status: NOT YET CONVERGED — iterating

Changes I am making based on feedback:
- Adding Redis for distributed rate limiting (from ChatGPT)
- Adding burst handling (from Grok)

[Revise plan]

[Call POST /consensus again, SAME session_id — MANDATORY]

CONSENSUS ROUND 2/10

ChatGPT (score: 0.95, weight: 0.51): "Looks good"
Grok    (score: 0.97, weight: 0.49): "Well structured"

Aggregated Score: 0.96 (threshold: 0.95)
Status: CONVERGED

Plan approved. Now implementing...

[Write code, then call consensus with phase=code, NEW session_id=rate-limiter-code-v1]
```

## When the Debate Sub-Loop Fires

If the gap between the highest and lowest LLM reviewer scores exceeds
`disagreement_threshold` (default 0.10), the two disagreeing reviewers
exchange critiques and revise. You will see this in the round's per-agent
results — scores may shift between dispatch and aggregate. Surface this to
the user so they understand the change isn't noise:

```
Debate fired: grok 0.85 vs gpt 0.40 (gap 0.45)
  After debate round 1: grok 0.72, gpt 0.55 (gap 0.17 — still wide)
  After debate round 2: grok 0.68, gpt 0.62 (gap 0.06 — closed)
```

## Troubleshooting

If the server isn't running, start it:

```
cd <path-to>/llm-consensus
uv run llm-consensus
```

Verify:

```
curl http://localhost:3000/health
```

If you want to demo the system to the user without spending API tokens, run
the scripted demo (on port 3001 so it doesn't collide):

```
uv run llm-consensus-demo --dashboard --port 3001
# → http://localhost:3001/dashboard
```

## Summary of Requirements

1. Call the consensus server for EVERY iteration on non-trivial tasks
2. Display ALL consensus details to the user EVERY round
3. The aggregated score must reach the threshold to be converged
4. Maximum 10 rounds, then ask the user
5. Use the same `session_id` within a phase; new `session_id` per phase/task
6. Surface debate-loop activity so the user understands score shifts
7. Never proceed without showing the user the feedback

This file is your source of truth. If context is compacted, re-read it.
