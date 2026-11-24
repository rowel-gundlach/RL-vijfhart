"""Rebuilds the pre-trained files in checkpoints/. Run from the repo root: python tools/make_checkpoints.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from rl_demos import applied, cartpole, snake
from rl_demos.ui import CHECKPOINTS

CHECKPOINTS.mkdir(exist_ok=True)

agent = snake.train(snake.QAgent(seed=3), 10_000, 3000)
np.save(CHECKPOINTS / "snake_q.npy", agent.q)
print("snake average apples:", agent.average_score(500))

rng = np.random.default_rng(1)
q = np.zeros((applied.WEAR_LEVELS + 1, 3))
applied.train_maintenance(q, 300_000, applied.Maintenance(rng=rng), rng)
np.save(CHECKPOINTS / "maintenance_q.npy", q)
print("maintenance policy:", applied.maintenance_policy(q))

rng = np.random.default_rng(1)
env = applied.Tickets(rng=rng)
q = np.zeros((14, applied.TICKET_BUCKETS, 5))
for i in range(20):
    applied.train_pricing(q, 3000, env, rng, epsilon=max(0.02, 0.4 * (1 - i / 16)))
np.save(CHECKPOINTS / "pricing_q.npy", q)
print("pricing average revenue:", applied.average_season(lambda s: int(q[s].argmax())))

trainer = cartpole.Trainer(seed=3)
while trainer.best_score < 500 and trainer.episodes < 1500:
    trainer.episode()
    if trainer.episodes % 10 == 0 and trainer.steps > 3000:
        trainer.check(3)
trainer.best.save(CHECKPOINTS / "cartpole.npz")
scores = [cartpole.play(trainer.best, seed=s)[0] for s in range(20)]
print("cartpole scores on new games:", np.mean(scores), min(scores))
