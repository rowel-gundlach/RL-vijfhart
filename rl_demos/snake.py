"""Demo 2: Snake played by a hand-written greedy rule, then by tabular Q-learning."""
import asyncio
import random

import anywidget
import numpy as np
import traitlets

from .ui import CHECKPOINTS, breathe, css, esm

UP, RIGHT, DOWN, LEFT = range(4)
MOVES = [(0, -1), (1, 0), (0, 1), (-1, 0)]
TURNS = [0, -1, 1]                       # straight, left, right
RULE_GRID = 10
LEARN_GRID = 10
WALL, DEAD_END, OPEN = range(3)          # what the snake sees in each direction
MILESTONES = (10, 100, 500, 1000, 5000, 10_000)  # the table is saved after these games, to look back at and play


class SnakeGame:
    def __init__(self, size, rng, snake=None, heading=UP, apples=()):
        self.size, self.rng, self.apples = size, rng, list(apples)
        middle = size // 2
        self.snake = list(snake) if snake else [(middle, middle), (middle, middle + 1)]
        self.heading = heading
        self.score = self.hunger = 0
        self.done = False
        self.apple = self._new_apple()

    def _new_apple(self):
        if self.apples:
            return self.apples.pop(0)
        free = [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) not in self.snake]
        return self.rng.choice(free) if free else None

    def blocked(self, cell, body=None):
        x, y = cell
        off_board = not (0 <= x < self.size and 0 <= y < self.size)
        return off_board or cell in (body if body is not None else self.snake[:-1])

    def room(self, cell, body):
        """WALL if the cell is blocked, DEAD_END if the space behind it is too small to fit the snake, else OPEN."""
        if self.blocked(cell, body):
            return WALL
        need, seen, todo = len(self.snake), {cell}, [cell]
        while todo and len(seen) < need:
            x, y = todo.pop()
            for dx, dy in MOVES:
                nxt = (x + dx, y + dy)
                if nxt not in seen and not self.blocked(nxt, body):
                    seen.add(nxt)
                    todo.append(nxt)
        return OPEN if len(seen) >= need else DEAD_END

    def ahead(self, heading):
        x, y = self.snake[0]
        dx, dy = MOVES[heading]
        return x + dx, y + dy

    def step(self, heading, apple_reward=10, death_reward=-10, step_reward=0):
        self.heading = heading
        head = self.ahead(heading)
        self.hunger += 1
        if self.blocked(head) or self.hunger > 2 * self.size ** 2:
            self.done = True
            return death_reward
        self.snake.insert(0, head)
        if head == self.apple:
            self.score += 1
            self.hunger = 0
            self.apple = self._new_apple()
            if self.apple is None:
                self.done = True
            return apple_reward
        self.snake.pop()
        return step_reward

    def frame(self, reward=0):
        return {"snake": list(self.snake), "apple": self.apple, "reward": reward, "score": self.score, "dead": self.done}


def towards_apple(game):
    """Headings that bring the snake closer to the apple, best first."""
    x, y = game.snake[0]
    ax, ay = game.apple
    options = []
    if ax > x: options.append((ax - x, RIGHT))
    if ax < x: options.append((x - ax, LEFT))
    if ay > y: options.append((ay - y, DOWN))
    if ay < y: options.append((y - ay, UP))
    return [heading for _, heading in sorted(options, reverse=True)] or [game.heading]


def greedy(game):
    """The rule: go for the apple, but never into something right in front of you."""
    for heading in towards_apple(game) + [game.heading, UP, RIGHT, DOWN, LEFT]:
        if not game.blocked(game.ahead(heading)):
            return heading
    return game.heading


def play_rule(strategy, seed=None, game=None):
    game = game or SnakeGame(RULE_GRID, random.Random(seed))
    frames = [game.frame()]
    while not game.done and game.apple is not None:
        frames.append(game.frame(game.step(strategy(game))))
    return frames


class QAgent:
    """One row per situation the snake can see, one number per move."""

    def __init__(self, size=LEARN_GRID, learning_rate=0.1, discount=0.9, apple_reward=10, death_reward=-10,
                 step_reward=-0.1, seed=None):
        self.size, self.alpha, self.gamma = size, learning_rate, discount
        self.rewards = dict(apple_reward=apple_reward, death_reward=death_reward, step_reward=step_reward)
        self.rng = random.Random(seed)
        self.q = np.zeros((3, 3, 3, 3, 3, 4, 3))
        self.games = 0

    @staticmethod
    def state(game):
        x, y = game.snake[0]
        ax, ay = game.apple if game.apple else (x, y)
        body = set(game.snake[:-1])
        looks = tuple(game.room(game.ahead((game.heading + turn) % 4), body) for turn in TURNS)
        return looks + (int(np.sign(ax - x)) + 1, int(np.sign(ay - y)) + 1, game.heading)

    def act(self, state, epsilon=0.0):
        if self.rng.random() < epsilon:
            return self.rng.randrange(3)
        row = self.q[state]
        best = row.max()
        return self.rng.choice([i for i in range(3) if row[i] == best])

    def train_game(self, epsilon):
        game = SnakeGame(self.size, self.rng)
        state = self.state(game)
        while not game.done:
            action = self.act(state, epsilon)
            reward = game.step((game.heading + TURNS[action]) % 4, **self.rewards)
            next_state = self.state(game)
            target = reward + (0 if game.done else self.gamma * self.q[next_state].max())
            self.q[state + (action,)] += self.alpha * (target - self.q[state + (action,)])
            state = next_state
        self.games += 1
        return game.score

    def play(self, record=True):
        game = SnakeGame(self.size, self.rng)
        frames = [game.frame()]
        while not game.done:
            action = self.act(self.state(game))
            reward = game.step((game.heading + TURNS[action]) % 4, **self.rewards)
            if record:
                frames.append(game.frame(reward))
        return frames if record else game.score

    def average_score(self, games=200):
        return float(np.mean([self.play(record=False) for _ in range(games)]))

    def epsilon(self, decay_games):
        return max(0.01, 1 - self.games / decay_games)


def train(agent, games, decay_games):
    for _ in range(games):
        agent.train_game(agent.epsilon(decay_games))
    return agent


class SnakeDemo(anywidget.AnyWidget):
    _esm = esm("snake.js")
    _css = css()
    view = traitlets.Dict().tag(sync=True)

    def __init__(self, apple_reward=10, death_reward=-10, step_reward=-0.1, learning_rate=0.1,
                 discount=0.9, games=10_000, show_every=2500, exploration_games=3000,
                 use_pretrained=False, seed=None):
        super().__init__()
        self.agent_settings = dict(learning_rate=learning_rate, discount=discount, apple_reward=apple_reward,
                                   death_reward=death_reward, step_reward=step_reward, seed=seed)
        self.games, self.show_every, self.exploration_games = games, show_every, exploration_games
        self.seed, self.use_pretrained = seed, use_pretrained
        self.task = None
        self.waiting = None               # (event loop, event) while training waits for the browser to show a game
        self._reset()
        self.on_msg(self._on_message)

    # JupyterLab can deliver each button press on another thread than the one that trains. asyncio is not
    # thread-safe, so anything that touches the training task is handed to that task's own event loop.
    def _stop(self):
        if self.task:
            self.task.get_loop().call_soon_threadsafe(self.task.cancel)
        self.task = None

    def _game_was_shown(self):
        if self.waiting:
            loop, event = self.waiting
            loop.call_soon_threadsafe(event.set)

    def _reset(self):
        self._stop()
        self.agent = QAgent(**self.agent_settings)
        self.curve = []
        self.snapshots = {}               # games played -> a copy of the table at that moment
        if self.use_pretrained:
            self.agent.q = np.load(CHECKPOINTS / "snake_q.npy")
            self.agent.games = self.games
            self.curve = [[self.games, round(self.agent.average_score(), 2)]]
            self.snapshots[self.games] = self.agent.q.copy()
        self._publish()

    def _publish(self, training=False):
        self.view = {"game": self.agent.games, "total": self.games, "curve": self.curve, "training": training,
                     "apple": self.agent_settings["apple_reward"], "death": self.agent_settings["death_reward"],
                     "moments": sorted(self.snapshots)}

    def label(self):
        games = self.agent.games
        if games < 500: return f"Game {games}: clueless"
        if games < self.exploration_games: return f"Game {games}: getting it"
        if games < self.games: return f"Game {games}: good, still improving"
        return f"Game {games}: expert"

    def _replay(self, frames, label, size):
        self.send({"type": "replay", "frames": frames, "label": label, "size": size})

    async def _show_agent(self):
        event = asyncio.Event()           # a new one each time: an event cannot move between event loops
        self.waiting = (asyncio.get_running_loop(), event)
        self._replay(self.agent.play(), self.label(), self.agent.size)
        try:
            await asyncio.wait_for(event.wait(), 120)
        except asyncio.TimeoutError:
            pass

    def milestones(self):
        return [m for m in MILESTONES if m < self.games] + [self.games]

    async def _train(self):
        await self._show_agent()
        while self.agent.games < self.games:
            games = self.agent.games
            upcoming = [m - games for m in self.milestones() if m > games]
            train(self.agent, min(50 - games % 50, *upcoming), self.exploration_games)
            games = self.agent.games
            if games in self.milestones():
                self.snapshots[games] = self.agent.q.copy()
            finished = games >= self.games
            if games % 500 == 0 or finished:
                self.curve.append([games, round(self.agent.average_score(60), 2)])
            self._publish(training=True)
            await breathe()
            if finished or games % max(50, self.show_every - self.show_every % 50) == 0:
                await self._show_agent()
        self.task = None
        self._publish()

    def _on_message(self, _, message, __):
        kind = message["type"]
        if kind == "shown":
            self._game_was_shown()
        elif kind == "reset":
            self._reset()
        elif kind == "train":
            if self.task:
                self._stop()
                self._publish()
            else:
                self.task = asyncio.ensure_future(self._train())
        elif kind == "play":
            mode = message["mode"]
            if mode == "greedy":
                self._replay(play_rule(greedy, self.seed), "Greedy: go for the apple, avoid crashes", RULE_GRID)
            elif message.get("moment") in self.snapshots:      # the snake as it was after that many games
                games = message["moment"]
                past = QAgent(**self.agent_settings)
                past.q, past.rng = self.snapshots[games], self.agent.rng
                self._replay(past.play(), f"The learning snake after {games:,} games of training", past.size)
            else:
                self._replay(self.agent.play(), self.label(), self.agent.size)


# Four situations for the live table, as real little boards with the snake heading up. The body runs head to tail.
SITUATIONS = [
    ("The apple is straight ahead", [(3, 3), (3, 4), (3, 5)], (3, 1)),
    ("The apple is to the left", [(3, 3), (3, 4), (3, 5)], (1, 3)),
    ("The apple is ahead, a wall to the left", [(0, 3), (0, 4), (0, 5)], (0, 1)),
    ("The apple is to the left, in a dead end",
     [(3, 3), (3, 4), (2, 4), (1, 4), (0, 4), (0, 3), (0, 2), (1, 2), (2, 2), (2, 1)], (1, 3)),
]
SITUATION_GRID = 7


def situation_game(body, apple):
    return SnakeGame(SITUATION_GRID, random.Random(0), snake=body, heading=UP, apples=[apple])


def situation_values(q, game):
    """The table's values for this picture, averaged over the four ways it can be turned (None = never tried)."""
    *looks, across, down, heading = QAgent.state(game)
    x, y, rows = across - 1, down - 1, []
    for turn in range(4):
        rows.append(q[tuple(looks) + (x + 1, y + 1, (heading + turn) % 4)])
        x, y = -y, x                     # a quarter turn clockwise
    seen = [row for row in rows if row.any()]
    return np.mean(seen, axis=0) if seen else None
