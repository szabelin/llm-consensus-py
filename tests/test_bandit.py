from llm_consensus.policy import bandit
from llm_consensus.state import BanditStats


def test_initialise_creates_zero_stats():
    b = bandit.initialise(["a", "b", "c"])
    assert all(s.pulls == 0 and s.reward_sum == 0.0 for s in b.values())


def test_cold_start_weights_are_uniform():
    b = bandit.initialise(["a", "b"])
    w = bandit.weights(b)
    assert w == {"a": 0.5, "b": 0.5}


def test_update_rewards_agents_close_to_aggregate():
    b = bandit.initialise(["a", "b"])
    # Aggregate = 0.8. Agent a said 0.8 (perfect), agent b said 0.5 (off by 0.3).
    new = bandit.update(b, {"a": 0.8, "b": 0.5}, aggregated=0.8)
    assert new["a"].reward_sum == 1.0  # 1 - |0.8 - 0.8|
    assert abs(new["b"].reward_sum - 0.7) < 1e-9  # 1 - |0.5 - 0.8|


def test_ucb_explores_unseen_arms_first():
    b = {"seen": BanditStats(pulls=5, reward_sum=4.5), "unseen": BanditStats()}
    w = bandit.weights(b)
    # Cold-start branch: any inf score forces uniform weights.
    assert w["seen"] == w["unseen"]


def test_ucb_favours_high_mean_when_all_seen():
    b = {
        "good": BanditStats(pulls=10, reward_sum=9.0),
        "bad": BanditStats(pulls=10, reward_sum=2.0),
    }
    w = bandit.weights(b)
    assert w["good"] > w["bad"]
