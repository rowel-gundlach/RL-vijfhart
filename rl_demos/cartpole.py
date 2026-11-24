"""Demo 7: a small DQN for CartPole, written in plain NumPy."""
import asyncio
from collections import deque

import anywidget
import gymnasium as gym
import numpy as np
import traitlets
from IPython.display import HTML

from .ui import CHECKPOINTS, breathe, css, esm

FALLEN = np.pi / 2 + 0.24                # the pole's angle when its tip lies on the track
NEURONS_DRAWN = 8                        # per hidden layer, in the replay


class Network:
    """4 numbers in, 2 values out, with two hidden layers."""

    def __init__(self, hidden=64, seed=None, inputs=4, outputs=2):
        rng = np.random.default_rng(seed)
        sizes = [inputs, hidden, hidden, outputs]
        self.weights = [rng.normal(0, np.sqrt(1 / a), (a, b)) for a, b in zip(sizes, sizes[1:])]
        self.biases = [np.zeros(b) for b in sizes[1:]]
        self.moments = [(np.zeros_like(p), np.zeros_like(p)) for p in self.weights + self.biases]
        self.updates = 0
        self.start = [p.copy() for p in self.weights + self.biases]    # to show "before training" later

    def untrained(self):
        """A copy of this network as it was before it learned anything."""
        net = Network(self.weights[0].shape[1])
        net.weights = [p.copy() for p in self.start[:len(self.weights)]]
        net.biases = [p.copy() for p in self.start[len(self.weights):]]
        return net

    def is_untrained(self):
        return all(np.array_equal(a, b) for a, b in zip(self.start, self.weights + self.biases))

    def forward(self, x, keep=False):
        layers = [x]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(0, x @ w + b)
            layers.append(x)
        out = x @ self.weights[-1] + self.biases[-1]
        return (out, layers) if keep else out

    def learn(self, states, actions, targets, rate=1e-3):
        """One Adam step that moves the chosen actions' values towards their targets."""
        out, layers = self.forward(states, keep=True)
        rows = np.arange(len(states))
        error = np.clip(out[rows, actions] - targets, -10, 10)
        signal = np.zeros_like(out)
        signal[rows, actions] = error
        grads_w, grads_b = [], []
        for layer, w in zip(reversed(layers), reversed(self.weights)):
            grads_w.insert(0, layer.T @ signal / len(states))
            grads_b.insert(0, signal.mean(axis=0))
            signal = (signal @ w.T) * (layer > 0)
        self.updates += 1
        for param, grad, (m, v) in zip(self.weights + self.biases, grads_w + grads_b, self.moments):
            m += 0.1 * (grad - m)
            v += 0.001 * (grad ** 2 - v)
            m_hat = m / (1 - 0.9 ** self.updates)
            v_hat = v / (1 - 0.999 ** self.updates)
            param -= rate * m_hat / (np.sqrt(v_hat) + 1e-8)

    def copy_from(self, other):
        self.weights = [w.copy() for w in other.weights]
        self.biases = [b.copy() for b in other.biases]

    def save(self, path):
        np.savez(path, **{f"w{i}": w for i, w in enumerate(self.weights)}, **{f"b{i}": b for i, b in enumerate(self.biases)})

    def load(self, path):
        data = np.load(path)
        self.weights = [data[f"w{i}"] for i in range(len(self.weights))]
        self.biases = [data[f"b{i}"] for i in range(len(self.biases))]


class Trainer:
    def __init__(self, discount=0.99, learning_rate=1e-3, hidden=64, batch=64, memory=20_000, sync_every=500, seed=None):
        self.net, self.target = Network(hidden, seed), Network(hidden, seed)
        self.best = Network(hidden, seed)
        self.gamma, self.rate, self.batch, self.sync_every = discount, learning_rate, batch, sync_every
        self.memory = deque(maxlen=memory)
        self.env = gym.make("CartPole-v1")
        self.rng = np.random.default_rng(seed)
        self.env.reset(seed=seed)
        self.steps = 0
        self.episodes = 0
        self.best_score = -1

    def episode(self):
        state, _ = self.env.reset()
        total, done = 0, False
        while not done:
            epsilon = max(0.02, 1 - self.steps / 6000)
            action = int(self.rng.integers(2)) if self.rng.random() < epsilon else int(self.net.forward(state).argmax())
            next_state, reward, ended, truncated, _ = self.env.step(action)
            done = ended or truncated
            self.memory.append((state, action, reward, next_state, ended))
            state = next_state
            total += reward
            self.steps += 1
            if len(self.memory) >= 500:
                self._replay()
            if self.steps % self.sync_every == 0:
                self.target.copy_from(self.net)
        self.episodes += 1
        return total

    def _replay(self):
        picks = self.rng.integers(len(self.memory), size=self.batch)
        batch = [self.memory[i] for i in picks]
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        ended = np.array([b[4] for b in batch])
        # The same rule as the table: reward + discounted best next value.
        # (The network picks the next action and the older target copy scores it, which is steadier.)
        next_actions = self.net.forward(next_states).argmax(axis=1)
        next_values = self.target.forward(next_states)[np.arange(self.batch), next_actions]
        targets = rewards + self.gamma * next_values * ~ended
        self.net.learn(states, actions, targets, self.rate)

    def check(self, games=5):
        """Keep the best network seen so far: training can get worse again."""
        score = float(np.mean([play(self.net, seed=9000 + i)[0] for i in range(games)]))
        if score >= self.best_score:
            self.best_score = score
            self.best.copy_from(self.net)
        return score


def play(net=None, seed=0):
    """Plays one game, returns (score, every state it went through). No network means random moves."""
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    state, _ = env.reset(seed=seed)
    states, total, done = [state], 0, False
    while not done:
        action = int(rng.integers(2)) if net is None else int(net.forward(state).argmax())
        state, reward, ended, truncated, _ = env.step(action)
        states.append(state)
        total += reward
        done = ended or truncated
    env.close()
    return total, states


def fall(state):
    """What happens after game over, which Gymnasium does not simulate: [cart position, pole angle] until it lies still."""
    env = gym.make("CartPole-v1").unwrapped
    x, speed, angle, spin = map(float, state)
    edge, half_pole, dt = env.x_threshold, env.length, env.tau
    path = []

    def hit_the_end():
        nonlocal x, speed, spin
        x = float(np.clip(x, -edge, edge))
        spin += 0.75 * speed / half_pole   # the cart stops dead, the pole keeps its momentum
        speed = 0.0

    if abs(x) >= edge:
        hit_the_end()
    for _ in range(400):
        braking = -3 * speed               # nobody pushes any more, so the cart rolls to a stop
        spin += dt * (env.gravity * np.sin(angle) - np.cos(angle) * braking) / (4 / 3 * half_pole)
        angle += dt * spin
        speed += dt * braking
        x += dt * speed
        if abs(x) > edge:
            hit_the_end()
        landed = abs(angle) >= FALLEN
        if landed:
            angle = float(np.copysign(FALLEN, angle))
            spin *= -0.25                  # a small bounce
        path.append([round(x, 4), round(float(angle), 4)])
        if landed and abs(spin) < 0.2 and abs(speed) < 0.05:
            break
    return path


def replay(net, seed=0):
    """Everything the animation needs: the game, what the network computed at each step, and the fall."""
    score, states = play(net, seed)
    states = np.array(states)
    out, (_, *hidden) = net.forward(states, keep=True)
    # Draw the neurons that matter most in this game: those that change a lot and weigh heavily on the next layer.
    picks = [np.sort(np.argsort(layer.std(axis=0) * np.abs(after).sum(axis=1))[-NEURONS_DRAWN:])
             for layer, after in zip(hidden, net.weights[1:])]
    index = [np.arange(4), *picks, np.arange(2)]
    env = gym.make("CartPole-v1").unwrapped
    x, _, angle, _ = states[-1]
    off_track, tipped = abs(x) > env.x_threshold, abs(angle) > env.theta_threshold_radians
    return {"score": int(score), "states": states.round(4).tolist(), "values": out.round(4).tolist(),
            "actions": out.argmax(axis=1).tolist(),
            "layers": [layer[:, pick].round(4).tolist() for layer, pick in zip(hidden, picks)],
            "weights": [w[np.ix_(a, b)].round(4).tolist() for w, a, b in zip(net.weights, index, index[1:])],
            "neurons": net.weights[0].shape[1],
            "reason": "off the track" if off_track else "past 12°" if tipped else None,
            "limit": env.theta_threshold_radians, "edge": env.x_threshold,
            "fall": fall(states[-1]) if off_track or tipped else []}


class BeforeAfter(anywidget.AnyWidget):
    """Side-by-side replay: the untrained network on the left, the trained one on the right."""
    _esm = esm("cartpole_movie.js")
    _css = css()
    view = traitlets.Dict().tag(sync=True)

    def __init__(self, trained, seed=0):
        super().__init__()
        self.view = {"before": replay(trained.untrained(), seed), "after": replay(trained, seed),
                     "untrained": trained.is_untrained()}


def before_after(trained, seed=0):
    return BeforeAfter(trained, seed)


def network_diagram(hidden=64):
    """The network that plays CartPole."""
    def circles(x, count, labels=None):
        top = 150 - (count - 1) * 24
        parts = []
        for i in range(count):
            y = top + i * 48
            parts.append(f'<circle cx="{x}" cy="{y}" r="15" fill="#2f6fde"/>')
            if labels:
                anchor, dx = ("end", -24) if x < 200 else ("start", 24)
                parts.append(f'<text x="{x + dx}" y="{y + 6}" text-anchor="{anchor}" font-size="17" fill="#1f2d3d">{labels[i]}</text>')
        return [150 - (count - 1) * 24 + i * 48 for i in range(count)], "".join(parts)

    columns = [(170, 4, ["cart position", "cart speed", "pole angle", "pole speed"]), (300, 6, None), (420, 6, None), (550, 2, ["push left", "push right"])]
    drawn = [circles(*c) for c in columns]
    lines = "".join(
        f'<line x1="{a[0]}" y1="{y1}" x2="{b[0]}" y2="{y2}" stroke="#d9e1ea"/>'
        for (a, (ys1, _)), (b, (ys2, _)) in zip(zip(columns, drawn), zip(columns[1:], drawn[1:]))
        for y1 in ys1 for y2 in ys2)
    network = (f'<svg viewBox="0 0 690 300" style="width:100%;max-width:640px">{lines}{"".join(d[1] for d in drawn)}'
               f'<text x="360" y="296" text-anchor="middle" font-size="16" fill="#5d6c7c">{hidden} neurons per hidden layer (only 6 drawn)</text></svg>')
    return HTML(f"""<div style="font:19px Segoe UI,sans-serif;color:#1f2d3d;background:#fff;padding:12px">
      <b>CartPole: a neural network</b><br>{network}
      <div style="font-size:21px"><b>It answers one question: "how good is each action here?"</b></div></div>""")


class CartPoleDemo(anywidget.AnyWidget):
    _esm = esm("cartpole.js")
    _css = css()
    view = traitlets.Dict().tag(sync=True)

    def __init__(self, games=600, discount=0.99, learning_rate=1e-3, hidden=64, use_pretrained=False, seed=None):
        super().__init__()
        self.settings = dict(discount=discount, learning_rate=learning_rate, hidden=hidden, seed=seed)
        self.games, self.use_pretrained = games, use_pretrained
        self.task = None
        self._reset()
        self.on_msg(self._on_message)

    @property
    def network(self):
        return self.trainer.best

    def _reset(self):
        if self.task:
            self.task.cancel()
        self.task = None
        self.trainer = Trainer(**self.settings)
        self.curve = []
        self.recent = deque(maxlen=20)
        self.done = self.use_pretrained
        if self.use_pretrained:
            self.trainer.best.load(CHECKPOINTS / "cartpole.npz")
        self._publish()

    def _publish(self, running=False):
        self.view = {"curve": self.curve, "game": self.trainer.episodes, "total": self.games, "running": running,
                     "done": self.done, "best": self.trainer.best_score, "memory": len(self.trainer.memory)}

    async def _train(self):
        while self.trainer.episodes < self.games and self.trainer.best_score < 500:
            self.recent.append(self.trainer.episode())
            if self.trainer.episodes % 10 == 0:
                self.curve.append([self.trainer.episodes, round(float(np.mean(self.recent)), 1)])
                if self.trainer.steps > 3000:
                    self.trainer.check(3)
                self._publish(running=True)
            await breathe()
        self.done = True
        self.task = None
        self._publish()

    def _on_message(self, _, message, __):
        if message["type"] == "reset":
            self._reset()
        elif message["type"] == "train":
            if self.task:
                self.task.cancel()
                self.task = None
                self._publish()
            else:
                self.task = asyncio.ensure_future(self._train())
