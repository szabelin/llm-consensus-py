"""UCB1 multi-armed bandit for weighting agent votes.

Each agent is an "arm". After every round we observe a reward in [0, 1] equal
to `1 - |agent_score - aggregated_score|` — agents whose verdicts track the
consensus are rewarded; outliers are penalised.

The UCB1 score for an arm is:
    mean_reward + sqrt(2 * ln(total_pulls) / arm_pulls)
The second term is the "exploration bonus": arms we've pulled rarely get a
boost so the policy doesn't get stuck on early winners. We normalise these
scores into weights that sum to 1 and use them to compute the next round's
aggregated score (instead of a plain average).
"""

from __future__ import annotations

import math

from llm_consensus.state import BanditStats


def ucb_score(stats: BanditStats, total_pulls: int) -> float:
    if stats.pulls == 0:
        return float("inf")  # unseen arms always explored first
    exploration = math.sqrt(2.0 * math.log(max(total_pulls, 1)) / stats.pulls)
    return stats.mean + exploration


def weights(bandit: dict[str, BanditStats]) -> dict[str, float]:
    """Convert UCB scores into a probability-style weight per agent."""
    total_pulls = sum(s.pulls for s in bandit.values())
    raw = {name: ucb_score(s, total_pulls) for name, s in bandit.items()}
    if any(math.isinf(v) for v in raw.values()):
        # Cold start: uniform weights so unseen agents get an equal vote.
        n = len(raw)
        return {k: 1.0 / n for k in raw}
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def update(
    bandit: dict[str, BanditStats],
    scores: dict[str, float],
    aggregated: float,
) -> dict[str, BanditStats]:
    """Return a new bandit dict with rewards recorded for this round."""
    out: dict[str, BanditStats] = {}
    for name, prior in bandit.items():
        observed = scores.get(name)
        if observed is None:
            out[name] = prior.model_copy()
            continue
        reward = max(0.0, 1.0 - abs(observed - aggregated))
        out[name] = BanditStats(
            pulls=prior.pulls + 1,
            reward_sum=prior.reward_sum + reward,
        )
    return out


def initialise(names: list[str]) -> dict[str, BanditStats]:
    return {n: BanditStats() for n in names}
