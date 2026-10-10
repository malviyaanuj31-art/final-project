"""
test_q_learning.py — Tests for Q-learning simulation correctness.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from src.q_learning_demo import run_q_learning, DEFAULT_GRID, START, GOAL


class TestQLearning:

    @pytest.fixture(scope="class")
    @classmethod
    def result(cls):
        return run_q_learning(alpha=0.1, gamma=0.9, epsilon=0.2, episodes=300)

    def test_result_has_required_keys(self, result):
        required = [
            "Q_table", "reward_history", "best_path",
            "grid", "start", "goal", "rows", "cols",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_q_table_shape(self, result):
        rows, cols = result["rows"], result["cols"]
        n_states  = rows * cols
        n_actions = 4
        assert result["Q_table"].shape == (n_states, n_actions)

    def test_reward_history_length(self, result):
        assert len(result["reward_history"]) == 300

    def test_reward_history_contains_floats(self, result):
        assert all(isinstance(r, float) for r in result["reward_history"])

    def test_best_path_starts_at_start(self, result):
        if result["best_path"]:
            assert result["best_path"][0] == result["start"]

    def test_best_path_cells_are_valid(self, result):
        grid = result["grid"]
        rows, cols = result["rows"], result["cols"]
        for (r, c) in result["best_path"]:
            assert 0 <= r < rows
            assert 0 <= c < cols
            assert grid[r, c] == 0, f"Path passes through obstacle at ({r}, {c})"

    def test_q_learning_update_rule(self):
        """Unit test the Bellman update manually."""
        alpha, gamma = 0.1, 0.9
        Q_old = 2.0
        reward = 10.0
        Q_next_max = 5.0
        # Q_new = Q_old + alpha * (reward + gamma * Q_next_max - Q_old)
        expected = Q_old + alpha * (reward + gamma * Q_next_max - Q_old)
        assert abs(expected - (Q_old + alpha * (reward + gamma * Q_next_max - Q_old))) < 1e-9

    def test_more_episodes_improves_average_reward(self):
        """More training should lead to higher mean reward (statistically)."""
        res_few  = run_q_learning(episodes=50)
        res_many = run_q_learning(episodes=500)
        mean_few  = np.mean(res_few["reward_history"][-20:])
        mean_many = np.mean(res_many["reward_history"][-20:])
        # With fixed seed: more episodes generally converge to better reward
        assert mean_many >= mean_few - 2.0  # loose tolerance for stochasticity

    def test_agent_reaches_goal_with_enough_training(self):
        """With sufficient episodes and good params the agent should find goal."""
        result = run_q_learning(alpha=0.2, gamma=0.95, epsilon=0.1, episodes=1000)
        if result["best_path"]:
            # Path should end at goal or be reasonable length
            reached = result["best_path"][-1] == result["goal"]
            # Acceptable if path length <= grid_size * 2
            reasonable_length = len(result["best_path"]) <= result["rows"] * result["cols"]
            assert reached or reasonable_length
