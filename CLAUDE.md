# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Tutorial notebooks for *Reinforcement Learning: An Overview* (Kevin P. Murphy, arXiv:2412.05265v5, Dec 2025).
One `.py` generator per chapter → produces its own `.ipynb` in `notebooks/`.

## Environment

```bash
uv sync
# Register venv as Jupyter kernel
uv run python -m ipykernel install --user --name=rl-overview --display-name "Python (rl-overview)"
```

## Commands

| Task | Command |
|------|---------|
| Generate one notebook | `uv run python ch01_introduction.py` |
| Generate all | `for f in ch0*.py; do uv run python $f; done` |
| Launch Jupyter | `uv run jupyter notebook notebooks/` |
| Run notebook headless | `uv run jupyter nbconvert --to notebook --execute notebooks/ch01_introduction.ipynb` |

## Architecture

Each `ch0N_*.py` is a self-contained generator: it builds `nbformat` cells (markdown + code) and writes `notebooks/ch0N_*.ipynb`. No shared imports between chapter files — helpers like `GridWorld` are inlined per file so any notebook runs independently.

Cell order in every notebook:
1. Markdown — chapter title + paper section reference
2. `%pip install -q ...` dependency cell
3. Alternating markdown (theory, LaTeX) / code (runnable implementations)

## Chapter Map

| Generator | Notebook | Key implementations |
|-----------|----------|---------------------|
| `ch01_introduction.py` | §1 | GridWorld env, MDP tables, random rollout |
| `ch02_value_based_rl.py` | §2 | Value/policy iteration, SARSA, Q-learning |
| `ch03_policy_based_rl.py` | §3 | REINFORCE, Actor-Critic, PPO clipping |
| `ch04_model_based_rl.py` | §4 | Dyna-Q (0/5/50 steps), MCTS, model-error |
| `ch05_multi_agent_rl.py` | §5 | LP Nash solver, fictitious play, ind. Q-learning |
| `ch06_llms_and_rl.py` | §6 | RLHF pipeline, Bradley-Terry RM, DPO, GRPO |
| `ch07_other_topics.py` | §7 | UCB1/Thompson bandit, distributional RL, BC, offline RL |

## Key Dependencies Per Chapter

- **numpy + matplotlib**: all
- **scipy**: Ch5 (`linprog` for Nash LP)
- **torch**: Ch3 (REINFORCE, Actor-Critic), Ch6 (reward model), Ch7 (behavior cloning)
- **gymnasium**: Ch3 (CartPole-v1); degrades gracefully if unavailable
