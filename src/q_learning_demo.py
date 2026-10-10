"""
q_learning_demo.py — Educational Q-Learning grid-world simulation.

An agent learns to navigate a small grid from Start → Goal while
avoiding obstacles. Demonstrates states, actions, rewards, Q-table,
and exploration vs exploitation.

This is an educational simulation.
It does not control real farming equipment or make validated farm decisions.
"""

from __future__ import annotations

import numpy as np
from typing import Any

RANDOM_SEED = 42


# ── grid world ──────────────────────────────────────────────────────────────────

# Default 6×6 grid layout:  0=free, 1=obstacle
DEFAULT_GRID = np.array([
    [0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0],
])

START = (0, 0)
GOAL  = (5, 5)

ACTIONS = [
    (-1,  0),  # up
    ( 1,  0),  # down
    ( 0, -1),  # left
    ( 0,  1),  # right
]
ACTION_NAMES = ["Up", "Down", "Left", "Right"]
N_ACTIONS = len(ACTIONS)


# ── Q-learning ──────────────────────────────────────────────────────────────────

def run_q_learning(
    alpha:    float = 0.1,
    gamma:    float = 0.9,
    epsilon:  float = 0.2,
    episodes: int   = 500,
    grid:     np.ndarray = DEFAULT_GRID,
    start:    tuple = START,
    goal:     tuple = GOAL,
) -> dict[str, Any]:
    """
    Run Q-learning on the grid-world.

    Parameters
    ----------
    alpha    : learning rate
    gamma    : discount factor
    epsilon  : exploration rate (ε-greedy)
    episodes : number of training episodes

    Returns
    -------
    dict with Q-table, reward history, best path and grid info.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    rows, cols = grid.shape
    n_states = rows * cols

    # Q-table: (state, action)
    Q = np.zeros((n_states, N_ACTIONS))

    def state_idx(r, c):
        return r * cols + c

    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols and grid[r, c] == 0

    reward_per_episode: list[float] = []

    for ep in range(episodes):
        r, c = start
        total_reward = 0.0
        max_steps = rows * cols * 4  # prevent infinite loops

        for _ in range(max_steps):
            s = state_idx(r, c)

            # ε-greedy action selection
            if rng.random() < epsilon:
                action_idx = rng.integers(N_ACTIONS)
            else:
                action_idx = int(np.argmax(Q[s]))

            dr, dc = ACTIONS[action_idx]
            nr, nc = r + dr, c + dc

            if not is_valid(nr, nc):
                # Hit wall or boundary — stay, negative reward
                reward  = -1.0
                nr, nc  = r, c
            elif (nr, nc) == goal:
                reward = 10.0
            else:
                reward = -0.05  # small step cost encourages shortest path

            ns = state_idx(nr, nc)
            # Bellman update
            Q[s, action_idx] += alpha * (
                reward + gamma * np.max(Q[ns]) - Q[s, action_idx]
            )

            total_reward += reward
            r, c = nr, nc

            if (r, c) == goal:
                break

        reward_per_episode.append(total_reward)

    # Extract best path using greedy policy
    best_path = _greedy_path(Q, grid, start, goal, rows, cols)

    return {
        "Q_table":          Q,
        "reward_history":   reward_per_episode,
        "best_path":        best_path,
        "grid":             grid,
        "start":            start,
        "goal":             goal,
        "rows":             rows,
        "cols":             cols,
        "action_names":     ACTION_NAMES,
        "params": {
            "alpha":    alpha,
            "gamma":    gamma,
            "epsilon":  epsilon,
            "episodes": episodes,
        },
    }


def _greedy_path(
    Q: np.ndarray,
    grid: np.ndarray,
    start: tuple,
    goal: tuple,
    rows: int,
    cols: int,
) -> list[tuple]:
    """Follow the greedy policy to extract the best learned path."""
    r, c    = start
    path    = [(r, c)]
    visited = set()
    max_steps = rows * cols

    def state_idx(r, c):
        return r * cols + c

    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols and grid[r, c] == 0

    for _ in range(max_steps):
        if (r, c) == goal:
            break
        s = state_idx(r, c)
        visited.add(s)
        action_idx = int(np.argmax(Q[s]))
        dr, dc = ACTIONS[action_idx]
        nr, nc = r + dr, c + dc

        if not is_valid(nr, nc) or state_idx(nr, nc) in visited:
            break  # stuck or looping
        r, c = nr, nc
        path.append((r, c))

    return path
