# RL Tutorial: Server Room Temperature Control

This repository contains a simple reinforcement learning tutorial using Stable Baselines3 to train an agent to control the temperature of a server room. The environment simulates a server room where the agent must balance cooling and heating actions to keep the temperature within safe limits.

## Getting Started

1. Clone the repository
2. Install UV: https://docs.astral.sh/uv/getting-started/installation/
3. Create a virtual environment and install dependencies: `uv sync`
4. Run the sample environment: `uv run ./custom_env.py`
5. Train the agent: `uv run ./train.py`
6. Evaluate the trained agent: `uv run ./evaluate.py`
