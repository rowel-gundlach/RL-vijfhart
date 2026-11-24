# RL Training Day

Browser-based demos for a one-day, non-technical introduction to reinforcement learning. Nothing to install: everything runs on mybinder.org.

[![Launch on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/YOUR-GITHUB-NAME/YOUR-REPO/HEAD?urlpath=lab)

After publishing this repo on GitHub (it must be public), replace `YOUR-GITHUB-NAME/YOUR-REPO` in the link above.

| Notebook | Demo | Live training |
|---|---|---|
| `01_train_your_dog.ipynb` | Rewards and the value table | None, you are the trainer |
| `02_snake_rules_vs_learning.ipynb` | Hand-written rules vs. Q-learning | About a minute, plus the games it shows |
| `03_booster.ipynb` | A booster yes or no, every day: pay now, benefit later | Seconds |
| `04_dynamic_pricing.ipynb` | What a ticket should cost today | Seconds |
| `05_predictive_maintenance.ipynb` | When to repair a machine | Seconds |
| `06_build_your_own_rl.ipynb` | Define your own states, actions, rewards and probabilities | Seconds |
| `07_deep_rl_cartpole.ipynb` | A small neural network replaces the table | Under a minute on a laptop |
| `08_how_chatgpt_uses_rl.ipynb` | Toy RLHF and reward hacking | Instant |

The exercises for all notebooks are in `RL_exercises.docx`: one table per notebook, with what to try and what to look for.

## For the presenter

- **Launch Binder before the session.** The first launch can take several minutes. Later launches of the same version are faster.
- In each notebook: *Run → Run All Cells*, then use the buttons in the demo.
- Binder sessions stop when idle. If one stops, launch again.
- Every notebook starts with a settings cell. Change a value, then run that cell and the demo cell again.
- If live training is slow, set `USE_PRETRAINED = True` in the settings cell (demos 2, 5 and 7). Demo 4 (pricing) trains in seconds and has no such setting.
- Pre-trained files only match the default settings of that demo.
- In notebook 7, run the before/after cell again after training has finished.
- The Atari Breakout and AlphaGo stories are slides only, not demos.

## Running it locally (optional)

```
pip install -r requirements.txt jupyterlab
jupyter lab
```

Use JupyterLab in the browser, as Binder does. In VS Code's notebook view the cells run but the demo may not appear.

## For maintainers

- `rl_demos/` holds the Python: rules, environments and learning. `rl_demos/assets/` holds the JavaScript and CSS that draw the demos.
- `python tools/make_checkpoints.py` rebuilds the pre-trained files. Do this after changing any environment.
- The tests (`tests/`) are kept on the presenter's computer, not in this repo. There, `pytest` checks that each demo still tells its story, for example that learning beats the fixed rule.
